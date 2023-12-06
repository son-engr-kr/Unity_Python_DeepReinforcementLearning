using System.Collections;
using System.Collections.Generic;
using System.Threading;
using TMPro;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.UIElements;

public class BallBalancingSystem : MonoBehaviour
{
    [SerializeField] Rigidbody BallRigidBody;
    [SerializeField] Rigidbody PlateRigidBody;
    [SerializeField] GameObject TargetVisualization;
    [SerializeField] float PlateAngularSpeed = 30f;
    [SerializeField] float DeltaTime;
    [SerializeField] Material TargetInMaterial;
    [SerializeField] Material TargetOutMaterial;
    System.Diagnostics.Process PythonProcess;
    bool ProcessAlive = false;
    Thread ReadThread;
    Thread WriteThread;
    float TargetThreshold = 0.1f;
    Label LabelSuccessRate;
    Label LabelRewardSum;
    float RewardSum = 0;
    Label LabelIterationCount;
    void Start()
    {
        var root = GetComponent<UIDocument>().rootVisualElement;
        LabelSuccessRate = root.Q<Label>("LabelSuccessRate");
        LabelRewardSum = root.Q<Label>("LabelRewardSum");
        LabelIterationCount = root.Q<Label>("LabelIterationCount");

        TargetVisualization.GetComponent<Renderer>().material = TargetOutMaterial;
        TargetVisualization.transform.localScale = Vector3.one * (BallRigidBody.transform.localScale.x + TargetThreshold * 2);
        TargetVisualization.GetComponent<SphereCollider>().isTrigger = true;

        DeltaTime = Time.deltaTime;

        TargetPosition = new Vector3(0, 0, 0.5f);

        PythonProcess = new System.Diagnostics.Process();

        PythonProcess.StartInfo.FileName = $"{Application.streamingAssetsPath}/python/.venv/Scripts/python.exe";
        PythonProcess.StartInfo.Arguments = $"{Application.streamingAssetsPath}/python/python_scripts/ball_balancing.py";

        PythonProcess.EnableRaisingEvents = true;


        //PythonProcess.StartInfo.CreateNoWindow = true;//true -> no window
        PythonProcess.StartInfo.UseShellExecute = false;
        PythonProcess.StartInfo.RedirectStandardOutput = true;
        PythonProcess.StartInfo.RedirectStandardInput = true;
        PythonProcess.StartInfo.RedirectStandardError = true;

        Debug.Log("Process start!");
        PythonProcess.Start();

        ProcessAlive = true;


        ReadThread = new Thread(ReadLoop);
        ReadThread.Start();

        WriteThread = new Thread(WriteLoop);
        WriteThread.Start();

    }


    void ReadLoop()
    {
        while (ProcessAlive)
        {
            while (simulationState == SimulationState.NEED_INITIALIZING)
            {
                Thread.Sleep(15);
            }
            string readLine = PythonProcess.StandardOutput.ReadLine();
            if (readLine == null || readLine == "")
            {
                Thread.Sleep(15);
                continue;
            }
            //Debug.Log($"python(all): {readLine}|{readLine.Length}");
            switch (readLine)
            {
                case "/inputRequest;modelPath":
                    {
                        PythonProcess.StandardInput.WriteLine($"{Application.streamingAssetsPath}/python/pytorch_models/ballbalancing_model_v2.pth");

                        break;
                    }
                case "/inputRequest;state":
                    {
                        //3+3+2+3+1 = 12
                        
                        PythonProcess.StandardInput.WriteLine($"{Vector3ToString(BallPosition)},{Vector3ToString(BallSpeed)},{PlateRX},{PlateRZ},{Vector3ToString(TargetPosition)},{PlateAngularSpeed}");

                        break;
                    }
                case "/inputRequest;reward":
                    {
                        PythonProcess.StandardInput.WriteLine($"{Reward}");
                        RewardSum += Reward;
                        break;
                    }
                case "/inputRequest;done":
                    {
                        bool done = simulationState == SimulationState.DONE;
                        //Debug.Log($"done:{done}");
                        PythonProcess.StandardInput.WriteLine($"{done}");
                        if (done)
                        {
                            Thread.Sleep(300);
                            simulationState = SimulationState.NEED_INITIALIZING;
                        }
                        break;
                    }
                case string ac when ac.Contains("/output;action"):
                    {
                        var splitString = ac.Split(";");
                        var action = int.Parse(splitString[2]);
                        switch (action)
                        {
                            case 0:
                                {
                                    PlateRX += PlateAngularSpeed * DeltaTime;
                                    break;
                                }
                            case 1:
                                {
                                    PlateRX -= PlateAngularSpeed * DeltaTime;
                                    break;
                                }
                            case 2:
                                {

                                    PlateRZ += PlateAngularSpeed * DeltaTime;
                                    break;
                                }
                            case 3:
                                {
                                    PlateRZ -= PlateAngularSpeed * DeltaTime;
                                    break;
                                }
                             case 4:
                                {

                                    break;
                                }

                        }
                        break;
                    }
                case string ac when ac.Contains("/errorOutput"):
                    {
                        var splitString = ac.Split(";");
                        var errorString = splitString[1];
                        Debug.Log($"python(filter-error): {errorString}");
                        break;
                    }
                case string ac when ac.Contains("/infoOutput"):
                    {
                        var splitString = ac.Split(";");
                        var infoString = splitString[1];
                        Debug.Log($"python(filter-info): {infoString}");
                        break;
                    }
                default:
                    {
                        //Debug.Log($"python(filter-nonoperation): {readLine}");
                        break;
                    }
            }
            Thread.Sleep(10);
        }
    }
    Vector3 CurrentPosition = Vector3.zero;
    void WriteLoop()
    {
        //while (ProcessAlive)
        //{
        //    PythonProcess.StandardInput.WriteLine($"{CurrentPosition}");
        //    Thread.Sleep(3000);
        //}
    }

    [Header("State")]
    [SerializeField] Vector3 BallPosition;
    [SerializeField] Vector3 BallSpeed;
    [SerializeField] float PlateRX;
    [SerializeField] float PlateRZ;
    [SerializeField] Vector3 TargetPosition;

    [Header("Reward")]
    [SerializeField] float Reward;

    enum SimulationState
    {
        NEED_INITIALIZING,
        RUNNING,
        DONE
    }
    [SerializeField] SimulationState simulationState = SimulationState.NEED_INITIALIZING;
    [SerializeField] long ContactStartTimeMillis;
    [SerializeField] long CurrentTime;
    [SerializeField] long EphisodStartTime;
    [SerializeField] bool PrevContacted = false;

    int SuccessCount = 0;
    int FailCount = 0;
    void Update()
    {
        LabelRewardSum.text = $"{RewardSum:n5}";
        TargetVisualization.transform.position = TargetPosition;
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
            simulationState = SimulationState.RUNNING;

            float xTarget = Random.Range(-0.4f, 0.4f);
            float zTarget = Random.Range(-0.4f, 0.4f);
            TargetPosition = new Vector3(xTarget,0.05f,zTarget);
            TargetVisualization.transform.position = TargetPosition;
            PrevContacted = false;
            EphisodStartTime = CurrentTime;
            if(SuccessCount + FailCount == 0)
            {
                LabelSuccessRate.text = "---%";
            }
            else
            {
                LabelSuccessRate.text = $"{SuccessCount / (float)(SuccessCount + FailCount) * 100f:n1}%";
            }
            LabelIterationCount.text = $"{SuccessCount + FailCount}";
        }
        else if (simulationState == SimulationState.RUNNING)
        {
            BallPosition = BallRigidBody.position;
            BallSpeed = BallRigidBody.velocity;
            PlateRigidBody.rotation = Quaternion.Euler(PlateRX, 0, PlateRZ);
            if(BallPosition.y < -1f)
            {
                Debug.Log("System: Done due to ball falling");
                Reward = -1f;
                simulationState = SimulationState.DONE;
                FailCount++;

            }
            else
            {
                float xDist = (TargetPosition.x - BallPosition.x);
                float zDist = (TargetPosition.z - BallPosition.z);
                float dist = Mathf.Sqrt(xDist * xDist + zDist * zDist);
                //Reward = Mathf.Exp(-(xDist * xDist + zDist * zDist)) * 0.1f;
                if (dist < TargetThreshold)
                {
                    if (!PrevContacted)
                    {
                        TargetVisualization.GetComponent<Renderer>().material = TargetInMaterial;
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
                    Reward = timePassedFromContact  * Mathf.Exp(-dist) / 1000f / 10f;
                    PrevContacted = true;
                }
                else
                {
                    if (PrevContacted)
                    {
                        TargetVisualization.GetComponent<Renderer>().material = TargetOutMaterial;
                    }

                    Reward = -Mathf.Exp(dist) / Mathf.Exp(2)/1000f;
                    PrevContacted = false;
                    if (CurrentTime - EphisodStartTime > 20 * 1000)
                    {
                        Debug.Log("System: Done due to TimeOver");
                        //Reward = -1f;
                        simulationState = SimulationState.DONE;
                        FailCount++;
                    }
                }
            }
        }
        CurrentTime += (long)(Time.deltaTime * 1000f);
    }
    string Vector3ToString(Vector3 vector3)
    {
        return $"{vector3.x},{vector3.y},{vector3.z}";
    }
}
