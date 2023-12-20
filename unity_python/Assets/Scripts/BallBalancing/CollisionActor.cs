using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CollisionActor : MonoBehaviour
{
    public Action<Collision> CollisionEntered;
    public Action<Collision> CollisionExited;
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }
    private void OnCollisionEnter(Collision collision)
    {
        CollisionEntered?.Invoke(collision);
    }
    private void OnCollisionExit(Collision collision)
    {
        CollisionExited?.Invoke(collision);

    }
}
