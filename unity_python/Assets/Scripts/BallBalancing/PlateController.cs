using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlateController : MonoBehaviour
{
    // Start is called before the first frame update
    Rigidbody RBody;
    void Start()
    {
        RBody = GetComponent<Rigidbody>();
    }

    [SerializeField] float XRot = 0;
    [SerializeField] float ZRot = 0;
    void Update()
    {
        if(Input.GetKey(KeyCode.UpArrow))
        {
            XRot += 0.1f;
        }
        else if (Input.GetKey(KeyCode.DownArrow))
        {
            XRot -= 0.1f;
        }

        if (Input.GetKey(KeyCode.RightArrow))
        {
            ZRot += 0.1f;
        }
        else if (Input.GetKey(KeyCode.LeftArrow))
        {
            ZRot -= 0.1f;
        }

        RBody.rotation = Quaternion.Euler(XRot, 0, ZRot);
        //transform.localRotation = Quaternion.Euler(XRot, 0, ZRot);

    }
}
