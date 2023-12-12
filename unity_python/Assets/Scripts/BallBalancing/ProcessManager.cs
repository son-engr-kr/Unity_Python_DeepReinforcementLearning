using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ProcessManager : MonoBehaviour
{
    bool ProcessOwner = true;
    System.Diagnostics.Process PythonProcess;

    void Start()
    {
        if (ProcessOwner)
        {
            PythonProcess = new System.Diagnostics.Process();

            PythonProcess.StartInfo.FileName = $"{Application.streamingAssetsPath}/python/.venv/Scripts/python.exe";
            PythonProcess.StartInfo.Arguments = $"{Application.streamingAssetsPath}/python/python_scripts/ball_balancing_training.py";

            PythonProcess.EnableRaisingEvents = true;
            //PythonProcess.StartInfo.CreateNoWindow = true;//true -> no window
            //PythonProcess.StartInfo.UseShellExecute = false;
            //PythonProcess.StartInfo.RedirectStandardOutput = true;
            //PythonProcess.StartInfo.RedirectStandardInput = true;
            //PythonProcess.StartInfo.RedirectStandardError = true;

            Debug.Log("Process start!");
            PythonProcess.Start();
        }
    }
    void ProcessWaitLoop()
    {

        if (ProcessOwner)
        {
            PythonProcess.WaitForExit();
        }

    }
}
