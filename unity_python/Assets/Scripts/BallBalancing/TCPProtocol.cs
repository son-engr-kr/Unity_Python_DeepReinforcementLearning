
using System.Runtime.InteropServices;
using System;

public class TCPProtocol
{
    static public byte[] StructToByte<T>(T st)
    {
        int structureSize = Marshal.SizeOf(st);
        byte[] byteArray = new byte[structureSize];// Builds byte array
        IntPtr memPtr = IntPtr.Zero;
        memPtr = Marshal.AllocHGlobal(structureSize);// Allocate some unmanaged memory
        Marshal.StructureToPtr(st, memPtr, true);// Copy struct to unmanaged memory
        Marshal.Copy(memPtr, byteArray, 0, structureSize);// Copies to byte array
        Marshal.FreeHGlobal(memPtr);
        return byteArray;

    }
    public static T ByteToStruct<T>(byte[] buffer) where T : struct
    {
        int size = Marshal.SizeOf(typeof(T));

        if (size > buffer.Length)
        {
            throw new Exception();
        }

        IntPtr ptr = Marshal.AllocHGlobal(size);
        Marshal.Copy(buffer, 0, ptr, size);
        T obj = (T)Marshal.PtrToStructure(ptr, typeof(T));
        Marshal.FreeHGlobal(ptr);
        return obj;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct HEADER
    {
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 4)]
        public byte[] DATA_TYPE;//JSON or BYTE
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)]
        public byte[] SUBJECT_CODE;
        [MarshalAs(UnmanagedType.I4)]
        public Int32 BODY_SIZE;
    }
    public interface PacketBase { }
    
    public enum SUBJECT_ENUM_8LEN
    {
        HTBT____,
        BB_STATE,
        BB_REWRD,
        BB_DONE_,
        BB_ACTN_,
        BB_EPRQS,
    }

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct HEARTBEAT : PacketBase
    {
        [MarshalAs(UnmanagedType.I8)]
        public Int64 SEND_UNIXTIME;
    }

    //<==================Ball Balancing=====================


    public class BALL_BALANCING_ACTION : PacketBase
    {
        public Int32 Action;
    }

    public class BALL_BALANCING_STATE : PacketBase
    {
        public float BallPositionX;
        public float BallPositionZ;

        public float BallSpeedX;
        public float BallSpeedZ;

        public float PlateRX;
        public float PlateRZ;

        public float TargetPositionX;
        public float TargetPositionZ;
    }

    public class BALL_BALANCING_REWARD : PacketBase
    {
        public float Reward;
    }
    public class BALL_BALANCING_DONE : PacketBase
    {
        public bool Done;
    }

    public class BALL_BALANCING_NEXT_EPISODE_REQUIRE_TO_UNITY : PacketBase
    {
        public int EPISODE_COUNT;
    }

    //!==================Ball Balancing=====================


}
