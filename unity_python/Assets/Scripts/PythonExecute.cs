using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using UnityEngine;
using static UnityEditor.Rendering.CameraUI;

public class PythonExecute : MonoBehaviour
{
    // Start is called before the first frame update
    System.Diagnostics.Process PythonProcess;
    Thread OutputThread;
    Thread InputThread;
    bool ProcessAlive = false;
    void Start()
    {
        PythonProcess = new System.Diagnostics.Process();

        PythonProcess.StartInfo.FileName = $"{Application.streamingAssetsPath}/python/.venv/Scripts/python.exe";
        PythonProcess.StartInfo.Arguments = $"{Application.streamingAssetsPath}/python/python_scripts/test.py";

        PythonProcess.EnableRaisingEvents = true;


        //PythonProcess.StartInfo.CreateNoWindow = true;//true -> no window
        PythonProcess.StartInfo.UseShellExecute = false;
        PythonProcess.StartInfo.RedirectStandardOutput = true ;
        PythonProcess.StartInfo.RedirectStandardError = true ;

        PythonProcess.OutputDataReceived += (object sender, System.Diagnostics.DataReceivedEventArgs e) =>
        {
            Debug.Log($"output ev: {e.Data}");
            if(e.Data == "/input")
            {
                PythonProcess.StandardInput.WriteLine($"{CurrentPosition}");
            }
        };

        PythonProcess.StartInfo.RedirectStandardInput = true;



        Debug.Log("Process start!");
        PythonProcess.Start();

        PythonProcess.BeginOutputReadLine();

        //Thread 쓰려면 아래 주석 해제하고 BeginOutputReadLine() 주석
        ProcessAlive = true;
        //OutputThread = new Thread(OutputReadLoop);
        //OutputThread.Start();


        //InputThread = new Thread(InputWriteLoop);
        //InputThread.Start();
    }
    void OutputReadLoop()
    {
        while (ProcessAlive)
        {
            string output = PythonProcess.StandardOutput.ReadLine();
            Debug.Log($"output: {output}");
            Thread.Sleep(10);
        }
    }
    Vector3 CurrentPosition = Vector3.zero;
    void InputWriteLoop()
    {
        while (ProcessAlive)
        {
            PythonProcess.StandardInput.WriteLine($"{CurrentPosition}");
            Thread.Sleep(3000);
        }
    }
    private void OnDestroy()
    {
        ProcessAlive = false;
        //PythonProcess.WaitForExit();
    }
    // Update is called once per frame
    void Update()
    {
        CurrentPosition = transform.position;
    }
}
