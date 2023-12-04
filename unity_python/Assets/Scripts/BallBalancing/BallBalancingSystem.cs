using System.Collections;
using System.Collections.Generic;
using System.Threading;
using TMPro;
using Unity.VisualScripting;
using UnityEngine;

public class BallBalancingSystem : MonoBehaviour
{
    [SerializeField] Rigidbody BallRigidBody;
    [SerializeField] Rigidbody PlateRigidBody;
    [SerializeField] float PlateAngularSpeed = 30f;
    [SerializeField] float DeltaTime;
    System.Diagnostics.Process PythonProcess;
    bool ProcessAlive = false;
    Thread ReadThread;
    Thread WriteThread;
    void Start()
    {
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
        /*
        PythonProcess.OutputDataReceived += (object sender, System.Diagnostics.DataReceivedEventArgs e) =>
        {
            Debug.Log($"python: {e.Data}");
            if (simulationState != SimulationState.NEED_INITIALIZING)
            {
                switch (e.Data)
                {
                    case "/inputRequest,state":
                        {
                            //3+3+2+3 = 
                            PythonProcess.StandardInput.WriteLine($"{Vector3ToString(BallPosition)},{Vector3ToString(BallSpeed)},{PlateRX},{PlateRZ},{Vector3ToString(TargetPosition)},{PlateAngularSpeed}");

                            break;
                        }
                    case "/inputRequest,reward":
                        {
                            PythonProcess.StandardInput.WriteLine($"{Reward}");

                            break;
                        }
                    case "/inputRequest,done":
                        {
                            bool done = simulationState == SimulationState.DONE;
                            PythonProcess.StandardInput.WriteLine($"{done}");
                            if (done)
                            {
                                simulationState = SimulationState.NEED_INITIALIZING;
                            }
                            break;
                        }
                    case string ac when ac.Contains("/output,action"):
                        {
                            var splitString = ac.Split(",");
                            var action = int.Parse(splitString[2]);
                            switch (action)
                            {
                                case 0:
                                    {
                                        PlateRX += PlateAngularSpeed * Time.deltaTime;
                                        break;
                                    }
                                case 1:
                                    {
                                        PlateRX -= PlateAngularSpeed * Time.deltaTime;
                                        break;
                                    }
                                case 2:
                                    {

                                        PlateRZ += PlateAngularSpeed * Time.deltaTime;
                                        break;
                                    }
                                case 3:
                                    {
                                        PlateRZ -= PlateAngularSpeed * Time.deltaTime;
                                        break;
                                    }

                            }
                            break;
                        }
                    default:
                        {
                            Debug.Log($"python(filter): {e.Data}");
                            break;
                        }
                }
            }
        };
        */





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
            Debug.Log($"python(all): {readLine}|{readLine.Length}");
            switch (readLine)
            {
                case "/inputRequest;state":
                    {
                        //3+3+2+3+1 = 12
                        
                        PythonProcess.StandardInput.WriteLine($"{Vector3ToString(BallPosition)},{Vector3ToString(BallSpeed)},{PlateRX},{PlateRZ},{Vector3ToString(TargetPosition)},{PlateAngularSpeed}");

                        break;
                    }
                case "/inputRequest;reward":
                    {
                        PythonProcess.StandardInput.WriteLine($"{Reward}");

                        break;
                    }
                case "/inputRequest;done":
                    {
                        bool done = simulationState == SimulationState.DONE;
                        Debug.Log($"done:{done}");
                        PythonProcess.StandardInput.WriteLine($"{done}");
                        if (done)
                        {
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

                        }
                        break;
                    }
                default:
                    {
                        Debug.Log($"python(filter-nonoperation): {readLine}");
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
    void Update()
    {
        if (simulationState == SimulationState.NEED_INITIALIZING)
        {
            PlateRX = 0f;
            PlateRZ = 0f;
            Reward = 0f;
            BallRigidBody.position = new Vector3(0, 0.3f, 0);
            BallRigidBody.velocity = new Vector3(0, 0f, 0);
            BallRigidBody.angularVelocity = new Vector3(0, 0f, 0);
            simulationState = SimulationState.RUNNING;
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
            }
            else
            {
                float xDist = (TargetPosition.x - BallPosition.x);
                float zDist = (TargetPosition.z - BallPosition.z);
                Reward = Mathf.Exp(-(xDist * xDist + zDist * zDist));
                if((xDist * xDist + zDist * zDist) < 0.0001f)
                {
                    Debug.Log("System: Done due to ball goal");

                    simulationState = SimulationState.DONE;
                }
            }
        }
    }
    string Vector3ToString(Vector3 vector3)
    {
        return $"{vector3.x},{vector3.y},{vector3.z}";
    }
}
