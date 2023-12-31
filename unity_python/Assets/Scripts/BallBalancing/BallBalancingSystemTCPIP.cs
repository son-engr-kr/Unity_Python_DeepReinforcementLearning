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
    float TargetThreshold = 0.1f;
    float RewardSum = 0;
    int TimeLimitMillis = 20 * 1000;

    int _collisionStack = 0;
    //python is server(unity can change setting value but python cannot(hard))
    TcpClientWrapper _tcpClientWrapper;

    void Start()
    {

        
        //===========test===============

        var jsonString = JsonUtility.ToJson(new BALL_BALANCING_REWARD() { Reward = 3.141592f, });
        Debug.Log($"JSON test1: {jsonString}");
        var convertedObject = JsonUtility.FromJson<TCPProtocol.BALL_BALANCING_REWARD>(jsonString);
        Debug.Log($"JSON test2: {convertedObject.Reward}");

        //!=========test=================


        //================initialize===================

        simulationState = SimulationState.NEED_REQUEST;
        PrevContacted = false;

        //=============================================




        

        TargetVisualization.GetComponent<Renderer>().material = TargetOutMaterial;
        TargetVisualization.transform.localScale = Vector3.one * (BallRigidBody.transform.localScale.x + TargetThreshold * 2);
        TargetVisualization.GetComponent<SphereCollider>().isTrigger = true;

        DeltaTime = Time.deltaTime;

        TargetVisualization.transform.localPosition = new Vector3(0, 0, 0.5f);

        //=====================contact====================

        var collisionActors = GetComponentsInChildren<CollisionActor>();
        foreach(var collisionActor in collisionActors)
        {
            collisionActor.CollisionEntered += (Collision collision) =>
            {
                if(collision.gameObject != PlateRigidBody.gameObject)
                {
                    _collisionStack++;
                }
                //if(collision.gameObject != PlateRigidBody.gameObject)
                //{
                //    _collisionStack++;
                //    Debug.Log($"collide: {name}-{collision.gameObject.name}");

                //}
                //foreach (ContactPoint c in collision.contacts)
                //{
                //    //Debug.Log(c.thisCollider.name);
                //    Debug.Log($"collision enter: {c.thisCollider.name}");
                //    if(c.thisCollider.gameObject != PlateRigidBody.gameObject)
                //    {
                //        _collisionStack++;
                //    }

                //}
            };
            collisionActor.CollisionExited += (Collision collision) =>
            {
                if (collision.gameObject != PlateRigidBody.gameObject)
                {
                    _collisionStack--;
                }
            };
        }
        //================================================



        _tcpClientWrapper = gameObject.AddComponent<TcpClientWrapper>();
        _tcpClientWrapper.IPAddress = "127.0.0.1";
        _tcpClientWrapper.PortNum = 11200;

        _tcpClientWrapper.StartTCPClient();
        _tcpClientWrapper.PacketReceiveEvent += (string subjectCode8, byte[] bodyBytes)=>{
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
                            if(simulationState == SimulationState.RUNNING)
                            {
                                ActionUpdated = true;
                            }
                            break;
                        }
                    case TCPProtocol.SUBJECT_ENUM_8LEN.BB_EPRQS:
                        {
                            var packetConverted = JsonUtility.FromJson<TCPProtocol.BALL_BALANCING_NEXT_EPISODE_REQUIRE_TO_UNITY>(Encoding.UTF8.GetString(bodyBytes));
                            //Debug.Log($"EP REQ!!- {packetConverted.EPISODE_COUNT}");
                            EphisodeCount = packetConverted.EPISODE_COUNT;

                            break;
                        }
                }
            }
        };
    }
    

    [Header("State")]
    [SerializeField] float PlateRX;
    [SerializeField] float PlateRZ;

    [Header("Reward")]
    [SerializeField] float Reward;

    enum SimulationState
    {
        NEED_REQUEST,
        NEED_INITIALIZING,
        RUNNING,
        DONE
    }
    [SerializeField] SimulationState simulationState;
    [SerializeField] long ContactStartTimeMillis;
    [SerializeField] long CurrentTime;
    [SerializeField] long EphisodStartTime;
    [SerializeField] bool PrevContacted;

    int PrevEphisodeCount = -1;
    int EphisodeCount = -1;
    int SuccessCount = 0;
    int FailCount = 0;

    bool ActionUpdated = false;
    float PrevDist;
    void Update()
    {
        //TargetVisualization.transform.position = TargetVisualization.transform.position;
        if(simulationState == SimulationState.NEED_REQUEST)
        {
            BallRigidBody.angularVelocity = new Vector3(0, 0f, 0);
            BallRigidBody.transform.localPosition = new Vector3(0, 0.3f, 0);
            BallRigidBody.constraints = RigidbodyConstraints.FreezeAll;
            PlateRigidBody.rotation = Quaternion.Euler(0, 0, 0);

        }
        if (PrevEphisodeCount != EphisodeCount)//NEED_REQUEST -> NEED_INITIALIZING
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
            BallRigidBody.transform.localPosition = new Vector3(0, 0.3f, 0);
            BallRigidBody.velocity = new Vector3(0, 0f, 0);
            BallRigidBody.angularVelocity = new Vector3(0, 0f, 0);
            BallRigidBody.constraints = RigidbodyConstraints.None;

            float xTarget = Random.Range(-0.4f, 0.4f);
            float zTarget = Random.Range(-0.4f, 0.4f);
            TargetVisualization.transform.localPosition = new Vector3(xTarget, 0.05f, zTarget);


            PrevDist = Mathf.Sqrt(xTarget* xTarget + zTarget * zTarget);

            PrevContacted = false;
            EphisodStartTime = CurrentTime;
            ActionUpdated = true;
            
            
            simulationState = SimulationState.RUNNING;
        }
        else if (simulationState == SimulationState.RUNNING)
        {
            PlateRigidBody.rotation = Quaternion.Euler(PlateRX, 0, PlateRZ);

            if (_collisionStack > 0)
            {
                Reward -= 0.1f;
            }

            if (BallRigidBody.transform.localPosition.y < -1f)
            {
                //Debug.Log("System: Done due to ball falling");
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
                    if (PrevContacted && timePassedFromContact > 5000)//5�� �̻�
                    {
                        //Debug.Log("System: Done due to ball goal");
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
                    if (CurrentTime - EphisodStartTime > 5 * 1000)
                    {
                        //Debug.Log("System: Done due to TimeOver");
                        //Reward = -1f;
                        simulationState = SimulationState.DONE;
                        FailCount++;
                    }
                }
                PrevDist = dist;
            }
        }

        if (ActionUpdated || simulationState == SimulationState.DONE)
        {
            if(simulationState == SimulationState.DONE)
            {
                simulationState = SimulationState.NEED_REQUEST;
            }
            _tcpClientWrapper.EnqueuePacket(TCPProtocol.SUBJECT_ENUM_8LEN.BB_DONE_.ToString(), new TCPProtocol.BALL_BALANCING_DONE()
            {
                Done = simulationState != SimulationState.RUNNING,
            });
            _tcpClientWrapper.EnqueuePacket(TCPProtocol.SUBJECT_ENUM_8LEN.BB_STATE.ToString(), new TCPProtocol.BALL_BALANCING_STATE()
            {
                BallPositionX = this.BallRigidBody.transform.localPosition.x,
                BallPositionZ = this.BallRigidBody.transform.localPosition.z,

                BallSpeedX = this.BallRigidBody.velocity.x,
                BallSpeedZ = this.BallRigidBody.velocity.z,

                PlateRX = this.PlateRX/10f,
                PlateRZ = this.PlateRZ/10f,

                TargetPositionX = this.TargetVisualization.transform.localPosition.x,
                TargetPositionZ = this.TargetVisualization.transform.localPosition.z,
            });
            _tcpClientWrapper.EnqueuePacket(TCPProtocol.SUBJECT_ENUM_8LEN.BB_REWRD.ToString(), new TCPProtocol.BALL_BALANCING_REWARD()
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
