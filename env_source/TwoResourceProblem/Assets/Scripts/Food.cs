using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Food : MonoBehaviour
{
    float objects_initial_range = 20.0f;
    float objects_initial_height = 1.0f;
    Rigidbody rb;

    // Start is called before the first frame update
    void Start()
    {
        rb = this.GetComponent<Rigidbody>();
    }

    // Update is called once per frame
    void Update()
    {
        if (transform.position.y < -1.0f)
        {
            transform.position = this.GetPos();
            rb.velocity = Vector3.zero;
        }
    }

    public Vector3 GetPos()
    {
        return new Vector3(Random.Range(-this.objects_initial_range, this.objects_initial_range),
                        this.objects_initial_height,
                        Random.Range(-this.objects_initial_range, this.objects_initial_range));
    }
}
