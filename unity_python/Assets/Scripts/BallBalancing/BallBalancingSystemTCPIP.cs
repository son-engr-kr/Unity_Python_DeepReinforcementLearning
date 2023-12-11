using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using UnityEngine;
using UnityEngine.UIElements;
using static TCPProtocol;

public class BallBalancingSystemTCPIP : MonoBehaviour
{
    [SerializeField] Rigidbody BallRigidBody;
    [SerializeField] Rigidbody PlateRigidBody;
    [SerializeField] GameObject TargetVisualization;
    float PlateDegree = 30f;
    [SerializeField] float DeltaTime;
    [SerializeField] Material TargetInMaterial;
    [SerializeField] Material TargetOutMaterial;
    System.Diagnostics.Process PythonProcess;
    float TargetThreshold = 0.1f;
    Label LabelSuccessRate;
    Label LabelRewardSum;
    float RewardSum = 0;
    Label LabelIterationCount;
    int TimeLimitMillis = 20 * 1000;

    //python is server(unity can change setting value but python cannot(hard))
    TCPClient PythonTCPClient;

    bool ProcessOwner = false;
    void Start()
    {
        //===========test===============

        var jsonString = JsonUtility.ToJson(new BALL_BALANCING_REWARD() { Reward = 3.141592f, });
        Debug.Log($"JSON test1: {jsonString}");
        var convertedObject = JsonUtility.FromJson<TCPProtocol.BALL_BALANCING_REWARD>(jsonString);
        Debug.Log($"JSON test2: {convertedObject.Reward}");

        //!=========test=================


        var root = GetComponent<UIDocument>().rootVisualElement;
        LabelSuccessRate = root.Q<Label>("LabelSuccessRate");
        LabelRewardSum = root.Q<Label>("LabelRewardSum");
        LabelIterationCount = root.Q<Label>("LabelIterationCount");

        TargetVisualization.GetComponent<Renderer>().material = TargetOutMaterial;
        TargetVisualization.transform.localScale = Vector3.one * (BallRigidBody.transform.localScale.x + TargetThreshold * 2);
        TargetVisualization.GetComponent<SphereCollider>().isTrigger = true;

        DeltaTime = Time.deltaTime;

        TargetVisualization.transform.position = new Vector3(0, 0, 0.5f);

        if (ProcessOwner)
        {
            PythonProcess = new System.Diagnostics.Process();

            PythonProcess.StartInfo.FileName = $"{Application.streamingAssetsPath}/python/.venv/Scripts/python.exe";
            PythonProcess.StartInfo.Arguments = $"{Application.streamingAssetsPath}/python/python_scripts/ball_balancing/ball_balancing_training.py";

            PythonProcess.EnableRaisingEvents = true;
            //PythonProcess.StartInfo.CreateNoWindow = true;//true -> no window
            //PythonProcess.StartInfo.UseShellExecute = false;
            //PythonProcess.StartInfo.RedirectStandardOutput = true;
            //PythonProcess.StartInfo.RedirectStandardInput = true;
            //PythonProcess.StartInfo.RedirectStandardError = true;

            Debug.Log("Process start!");
            PythonProcess.Start();
        }



        PythonTCPClient = gameObject.AddComponent<TCPClient>();
        PythonTCPClient.IPAddress = "127.0.0.1";
        PythonTCPClient.PortNum = 11200;

        PythonTCPClient.StartTCPClient();
        PythonTCPClient.PacketReceiveEvent += (string subjectCode8, byte[] bodyBytes)=>{
            TCPProtocol.SUBJECT_ENUM_8LEN packetSubjectEnum;
            var success = System.Enum.TryParse(subjectCode8, out packetSubjectEnum);
            //Debug.Log($"TCP RECV: {subjectCode8} - {Encoding.UTF8.GetString(bodyBytes)}");
            if (success)
            {
                switch (packetSubjectEnum)
                {
                    case TCPProtocol.SUBJECT_ENUM_8LEN.BB_ACTN_:
                        {
                            //var packetConverted = TCPProtocol.ByteToStruct<TCPProtocol.BALL_BALANCING_ACTION>(bodyBytes);
                            var packetConverted = JsonUtility.FromJson<TCPProtocol.BALL_BALANCING_ACTION>(Encoding.UTF8.GetString(bodyBytes));
                            int action = packetConverted.Action;
                            switch (action)
                            {
                                case 0:
                                    {
                                        PlateRX += PlateDegree * DeltaTime;
                                        break;
                                    }
                                case 1:
                                    {
                                        PlateRX -= PlateDegree * DeltaTime;
                                        break;
                                    }
                                case 2:
                                    {

                                        PlateRZ += PlateDegree * DeltaTime;
                                        break;
                                    }
                                case 3:
                                    {
                                        PlateRZ -= PlateDegree * DeltaTime;
                                        break;
                                    }
                                case 4:
                                    {

                                        break;
                                    }

                            }
                            ActionUpdated = true;
                            break;
                        }
                    case TCPProtocol.SUBJECT_ENUM_8LEN.BB_EPRQS:
                        {
                            var packetConverted = JsonUtility.FromJson<TCPProtocol.BALL_BALANCING_NEXT_EPISODE_REQUIRE_TO_UNITY>(Encoding.UTF8.GetString(bodyBytes));
                            Debug.Log($"EP REQ!!- {packetConverted.EPISODE_COUNT}");
                            EphisodeCount = packetConverted.EPISODE_COUNT;

                            break;
                        }
                }
            }
        };
    }
    void ProcessWaitLoop()
    {

        if (ProcessOwner)
        {
            PythonProcess.WaitForExit();
        }

    }

    [Header("State")]
    [SerializeField] float PlateRX;
    [SerializeField] float PlateRZ;

    [Header("Reward")]
    [SerializeField] float Reward;

    enum SimulationState
    {
        NEED_INITIALIZING,
        RUNNING,
        DONE
    }
    [SerializeField] SimulationState simulationState = SimulationState.DONE;
    [SerializeField] long ContactStartTimeMillis;
    [SerializeField] long CurrentTime;
    [SerializeField] long EphisodStartTime;
    [SerializeField] bool PrevContacted = false;

    int PrevEphisodeCount = -1;
    int EphisodeCount = -1;
    int SuccessCount = 0;
    int FailCount = 0;

    bool ActionUpdated = false;
    float PrevDist;
    void Update()
    {
        LabelRewardSum.text = $"{RewardSum:n5}";
        TargetVisualization.transform.position = TargetVisualization.transform.position;
        if(PrevEphisodeCount != EphisodeCount)
        {
            PrevEphisodeCount = EphisodeCount;
            simulationState = SimulationState.NEED_INITIALIZING;
        }
        if (simulationState == SimulationState.NEED_INITIALIZING)
        {
            TargetVisualization.GetComponent<Renderer>().material = TargetOutMaterial;
            PlateRX = 0f;
            PlateRZ = 0f;
            RewardSum = 0f;
            Reward = 0f;
            BallRigidBody.position = new Vector3(0, 0.3f, 0);
            BallRigidBody.velocity = new Vector3(0, 0f, 0);
            BallRigidBody.angularVelocity = new Vector3(0, 0f, 0);

            float xTarget = Random.Range(-0.4f, 0.4f);
            float zTarget = Random.Range(-0.4f, 0.4f);
            TargetVisualization.transform.position = new Vector3(xTarget, 0.05f, zTarget);


            PrevDist = Mathf.Sqrt(xTarget* xTarget + zTarget * zTarget);

            PrevContacted = false;
            EphisodStartTime = CurrentTime;
            ActionUpdated = true;
            if (SuccessCount + FailCount == 0)
            {
                LabelSuccessRate.text = "---%";
            }
            else
            {
                LabelSuccessRate.text = $"{SuccessCount / (float)(SuccessCount + FailCount) * 100f:n1}%";
            }
            LabelIterationCount.text = $"{SuccessCount + FailCount}";
            
            simulationState = SimulationState.RUNNING;
        }
        else if (simulationState == SimulationState.RUNNING)
        {
            PlateRigidBody.rotation = Quaternion.Euler(PlateRX, 0, PlateRZ);
            if (BallRigidBody.position.y < -1f)
            {
                Debug.Log("System: Done due to ball falling");
                Reward -= 1f;
                simulationState = SimulationState.DONE;
                FailCount++;

            }
            else
            {
                float xDist = (TargetVisualization.transform.position.x - BallRigidBody.position.x);
                float zDist = (TargetVisualization.transform.position.z - BallRigidBody.position.z);
                float dist = Mathf.Sqrt(xDist * xDist + zDist * zDist);
                Reward += (TargetThreshold-dist)/10f;
                if (dist < TargetThreshold)
                {
                    if (!PrevContacted)
                    {
                        TargetVisualization.GetComponent<Renderer>().material = TargetInMaterial;
                        //Reward += 1f;
                    }
                    var timePassedFromContact = CurrentTime - ContactStartTimeMillis;
                    if (PrevContacted && timePassedFromContact > 5000)//5√  ¿ÃªÛ
                    {
                        Debug.Log("System: Done due to ball goal");
                        simulationState = SimulationState.DONE;
                        SuccessCount++;
                    }
                    if (!PrevContacted)
                    {
                        ContactStartTimeMillis = CurrentTime;
                    }
                    //Reward += timePassedFromContact * Mathf.Exp(-dist) / 1000f / 1000f;
                    PrevContacted = true;
                }
                else
                {
                    if (PrevContacted)
                    {
                        TargetVisualization.GetComponent<Renderer>().material = TargetOutMaterial;
                        //Reward -= 1f;
                    }

                    //Reward -= Mathf.Exp(dist) / Mathf.Exp(2) / 1000f;
                    PrevContacted = false;
                    if (CurrentTime - EphisodStartTime > 20 * 1000)
                    {
                        Debug.Log("System: Done due to TimeOver");
                        //Reward = -1f;
                        simulationState = SimulationState.DONE;
                        FailCount++;
                    }
                }
                PrevDist = dist;
            }
        }

        if (ActionUpdated)
        {
            PythonTCPClient.EnqueuePacket(TCPProtocol.SUBJECT_ENUM_8LEN.BB_DONE_.ToString(), new TCPProtocol.BALL_BALANCING_DONE()
            {
                Done = simulationState != SimulationState.RUNNING,
            });
            PythonTCPClient.EnqueuePacket(TCPProtocol.SUBJECT_ENUM_8LEN.BB_STATE.ToString(), new TCPProtocol.BALL_BALANCING_STATE()
            {
                BallPositionX = this.BallRigidBody.position.x,
                BallPositionZ = this.BallRigidBody.position.z,

                BallSpeedX = this.BallRigidBody.velocity.x,
                BallSpeedZ = this.BallRigidBody.velocity.z,

                PlateRX = this.PlateRX,
                PlateRZ = this.PlateRZ,

                TargetPositionX = this.TargetVisualization.transform.position.x,
                TargetPositionZ = this.TargetVisualization.transform.position.z,
            });
            PythonTCPClient.EnqueuePacket(TCPProtocol.SUBJECT_ENUM_8LEN.BB_REWRD.ToString(), new TCPProtocol.BALL_BALANCING_REWARD()
            {
                Reward = Reward,
            });
            RewardSum += Reward;
            Reward = 0;
            ActionUpdated = false;
        }

        CurrentTime += (long)(Time.deltaTime * 1000f);
    }
    string Vector3ToString(Vector3 vector3)
    {
        return $"{vector3.x},{vector3.y},{vector3.z}";
    }
}
