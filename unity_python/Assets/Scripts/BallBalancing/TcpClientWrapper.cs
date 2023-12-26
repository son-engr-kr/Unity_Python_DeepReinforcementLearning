using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using UnityEngine;
using UnityEngine.SocialPlatforms;

public class TcpClientWrapper : MonoBehaviour
{
    private TcpClient _clientSocket;
    NetworkStream _clientStream;
    public string IPAddress;
    public int PortNum;

    Mutex _socketMutex = new();
    private Thread ConnectionThread;
    private Thread ReceiveThread;
    private Thread SendThread;
    private Thread HeartbeatThread;
    public Action<TCPEvent> SocketConnectionEvent;
    public Action<string> PacketSendEventForLogging;
    public Action<string> PacketReceiveEventForLogging;
    public Action<string> TCPActionLogging;

    //public Action<string, TCPProtocol.PacketBase> PacketReceiveEvent;
    public Action<string, byte[]> PacketReceiveEvent;
    bool _runClient = false;
    public void StartTCPClient()
    {
        _runClient = true;

        ConnectionThread = new Thread(new ThreadStart(TCPSocketConnectLoop));
        ConnectionThread.Start();

        ReceiveThread = new Thread(new ThreadStart(ReceiveLoop));
        ReceiveThread.Start();

        SendThread = new Thread(new ThreadStart(SendLoop));
        SendThread.Start();

        HeartbeatThread = new Thread(HeartbeatLoop);
        HeartbeatThread.Start();


    }
    void SocketClose()
    {
        _clientStream?.Close();
        _clientStream = null;
        _clientSocket?.Close();
        _clientSocket = null;

    }
    public void StopTCPClient()
    {
        _runClient = false;
        SocketConnectionEvent?.Invoke(TCPEvent.NORMAL_CLOSED);

    }
    private void OnDestroy()
    {
        StopTCPClient();
    }
    void TCPConnect(out TcpClient socket, string ip, int port)
    {
        TcpClient tcp = new TcpClient();
        IAsyncResult ar = tcp.BeginConnect(ip, port, null, null);
        System.Threading.WaitHandle wh = ar.AsyncWaitHandle;
        try
        {
            if (!ar.AsyncWaitHandle.WaitOne(TimeSpan.FromSeconds(3), false))
            {
                tcp.Close();
                throw new TimeoutException();
            }

            tcp.EndConnect(ar);
            socket = tcp;
        }
        finally
        {
            wh.Close();
        }
    }
    long LastHeartbitDatetime;
    void TCPSocketConnectLoop()
    {
        while (_runClient)
        {

            _socketMutex.WaitOne();

            if (_clientSocket == null)
            {
                Thread.Sleep(500);
                try
                {
                    TCPConnect(out _clientSocket, IPAddress, PortNum);
                    LastHeartbitDatetime = TimeManager.GetRealNowTimeMillis();
                    if (_clientSocket != null && _clientSocket.Connected)
                    {
                        _clientStream = _clientSocket.GetStream();
                        SocketConnectionEvent?.Invoke(TCPEvent.CONNECTED);
                    }

                }
                catch (TimeoutException te)
                {
                    Debug.Log($"TCP: timeout! {te}");
                    SocketClose();
                    SocketConnectionEvent?.Invoke(TCPEvent.TIMEOUT_ERR);
                }
                catch (Exception ex)
                {
                    Debug.Log($"TCP: TCP fail, exception: {ex}");
                    SocketClose();
                    SocketConnectionEvent?.Invoke(TCPEvent.CONNECTION_FAIL);
                }
                finally
                {
                    if (!_runClient)
                    {
                        SocketConnectionEvent?.Invoke(TCPEvent.NORMAL_CLOSED);
                    }
                }
            }
            else
            {
                if (DateTimeOffset.Now.ToUnixTimeMilliseconds() - LastHeartbitDatetime > 5 * 1000L)
                {
                    
                }
            }
            _socketMutex.ReleaseMutex();


            Thread.Sleep(1000);
        }
        _socketMutex.WaitOne();
        SocketClose();
        _socketMutex.ReleaseMutex();
    }
    void HeartbeatLoop()
    {

        while (_runClient && _clientStream != null)
        {
            EnqueuePacket(TCPProtocol.SUBJECT_ENUM_8LEN.HTBT____.ToString(), new TCPProtocol.HEARTBEAT() { SEND_UNIXTIME = TimeManager.GetRealNowTimeMillis() });

            Thread.Sleep(1000);
        }

    }
    Mutex SendQueueMutex = new Mutex();
    Queue<KeyValuePair<string, TCPProtocol.PacketBase>> SendPackets = new Queue<KeyValuePair<string, TCPProtocol.PacketBase>>();
    public void EnqueuePacket(string subject, TCPProtocol.PacketBase bodyStruct)
    {
        SendQueueMutex.WaitOne();

        SendPackets.Enqueue(new KeyValuePair<string, TCPProtocol.PacketBase>(subject,bodyStruct));

        SendQueueMutex.ReleaseMutex();

    }
    void SendLoop()
    {
        Thread.Sleep(100);
        while (_runClient)
        {
            SendQueueMutex.WaitOne();
            KeyValuePair<string, TCPProtocol.PacketBase> packetBase;
            string subjectString = "";
            TCPProtocol.PacketBase packetStructureBase = null;
            if (SendPackets.Count > 0)
            {
                packetBase = SendPackets.Dequeue();
                subjectString = packetBase.Key;
                packetStructureBase = packetBase.Value;
            }
            SendQueueMutex.ReleaseMutex();

            if (packetStructureBase != null && subjectString.Length == 8)
            {
                JsonPacketSend(subjectString, packetStructureBase);
            }
            Thread.Sleep(5);
        }
    }
    void StructPacketSend(string SubjectCode, TCPProtocol.PacketBase bodyPacket)
    {
        var bodyBytes = TCPProtocol.StructToByte(bodyPacket);
        var headStruct = new TCPProtocol.HEADER()
        {
            DATA_TYPE = Encoding.UTF8.GetBytes("BYTE"),
            SUBJECT_CODE = Encoding.UTF8.GetBytes(SubjectCode),
            BODY_SIZE = bodyBytes.Length,


        };
        var headBytes = TCPProtocol.StructToByte(headStruct);
        _socketMutex.WaitOne();
        SendByteBuffer(headBytes.Concat(bodyBytes).ToArray(), _clientStream);
        _socketMutex.ReleaseMutex();
    }
    void JsonPacketSend(string SubjectCode, TCPProtocol.PacketBase bodyPacket)
    {
        var bodyBytes = Encoding.UTF8.GetBytes(JsonUtility.ToJson(bodyPacket));
        var headStruct = new TCPProtocol.HEADER()
        {
            DATA_TYPE = Encoding.UTF8.GetBytes("JSON"),
            SUBJECT_CODE = Encoding.UTF8.GetBytes(SubjectCode),
            BODY_SIZE = bodyBytes.Length,
        };
        var headBytes = TCPProtocol.StructToByte(headStruct);
        _socketMutex.WaitOne();
        SendByteBuffer(headBytes.Concat(bodyBytes).ToArray(), _clientStream);
        _socketMutex.ReleaseMutex();
    }
    private void SendByteBuffer(Byte[] buffer, NetworkStream stream)
    {
        try
        {

            if (stream.CanWrite)
            {
                //Debug.Log($"TCP: write:{Encoding.UTF8.GetString(buffer)}");

                stream.Write(buffer, 0, buffer.Length);
            }

        }
        catch (Exception e)
        {
            Debug.Log($"TCP: Error when sending byte buffer:{e.Message}, packet:{Encoding.UTF8.GetString(buffer)}");

            SocketClose();

            SocketConnectionEvent?.Invoke(TCPEvent.SEND_ERR);

        }


    }

    Mutex ReceiveQueueMutex = new Mutex();
    Queue<KeyValuePair<string, byte[]>> ReceivedPacketQueue = new Queue<KeyValuePair<string, byte[]>> ();
    void ReceiveLoop()
    {

        Thread.Sleep(100);
        while (_runClient)
        {
            _socketMutex.WaitOne();
            if (_clientSocket != null)
            {
                try
                {
                    if (_clientStream.DataAvailable)
                    {
                        var headStruct = new TCPProtocol.HEADER();
                        Byte[] headByte = new Byte[Marshal.SizeOf(headStruct)];
                        _clientStream.Read(headByte, 0, headByte.Length);
                        headStruct = TCPProtocol.ByteToStruct<TCPProtocol.HEADER>(headByte);
                        //Debug.Log($"TCP: Reveived: header, length: {headByte.Length} | headStruct.BODY_SIZE: {headStruct.BODY_SIZE}");

                        Byte[] bodyByte = new Byte[headStruct.BODY_SIZE];//ETX±îÁö
                        _clientStream.Read(bodyByte, 0, bodyByte.Length);
                        //Debug.Log($"TCP: Reveived: body, length: {bodyByte.Length} | {Encoding.Default.GetString(bodyByte)}");


                        ReceiveQueueMutex.WaitOne();
                        ReceivedPacketQueue.Enqueue(new KeyValuePair<string, byte[]>(Encoding.Default.GetString(headStruct.SUBJECT_CODE), bodyByte));
                        ReceiveQueueMutex.ReleaseMutex();
                    }

                }
                catch (System.Threading.ThreadAbortException tex)
                {
                    Debug.Log($"TCP receive thread aborted");
                }
                catch (Exception ex)
                {
                    Debug.Log($"TCP socket disconnected unnormally : {ex}");

                }

            }
            _socketMutex.ReleaseMutex();

            Thread.Sleep(5);
        }

    }
    private void Update()
    {
        ReceiveQueueMutex.WaitOne();
        if (ReceivedPacketQueue.Count > 0)
        {
            var packetKeyValue = ReceivedPacketQueue.Dequeue();
            string subjectCode8 = packetKeyValue.Key;
            byte[] bodyBytes = packetKeyValue.Value;
            PacketReceiveEvent?.Invoke(subjectCode8, bodyBytes);
        }
        ReceiveQueueMutex.ReleaseMutex();

    }

}
