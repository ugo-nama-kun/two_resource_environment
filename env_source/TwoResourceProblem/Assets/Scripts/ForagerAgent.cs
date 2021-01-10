using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;


public class ForagerAgent : Agent
{
    private int NUM_RESOURCES = 2;

    private float forward_speed = 5f;
    private float rotation_angle_speed = 200f;
    private float resource_decreasing_rate = 0.2f;  // [unit/sec]
    private CharacterController character;
    private float vertial_velocity;
    private bool taking_eat_behavior;
    private float resource_limit = 15f; // Episode Terminate if the agent exceed this limit

    // Resource Parameters
    public GameObject prefab_red;
    public GameObject prefab_blue;
    private float food_red_increase = 3.0f;
    private float food_blue_increase = 3.0f;
    private int num_resource_red = 50;
    private int num_resource_blue = 50;
    private float[] resource_levels;
    private GameObject[] objects_red;
    private GameObject[] objects_blue;
    private float objects_initial_height= 3.0f;
    private float objects_initial_range = 20.0f;

    // Monitor
    [DebugGUIPrint, DebugGUIGraph(max: 15f, min: -15f, group: 1, r: 1, g: 0.4f, b: 0.4f)]
    float resource_red;
    [DebugGUIPrint, DebugGUIGraph(max: 15f, min: -15f, group: 1, r: 0.4f, g: 0.4f, b: 1)]
    float resource_blue;

    private void Awake()
    {
        Physics.autoSimulation = false;
        QualitySettings.vSyncCount = 0;
        Application.targetFrameRate = (int)(60.0 * Time.timeScale);
    }

    private void OnDestroy()
    {
        Physics.autoSimulation = true;
    }

    // Start is called before the first frame update
    void Start()
    {
        print("Start");
        this.resource_levels = new float[this.NUM_RESOURCES];
        this.character = gameObject.GetComponent<CharacterController>();
        this.taking_eat_behavior = false;

        // Generate Food Objects
        this.objects_red = new GameObject[this.num_resource_red];
        for (int i = 0; i < this.num_resource_red; i++)
        {
            GameObject obj = Instantiate(prefab_red) as GameObject;
            objects_red[i] = obj;
            objects_red[i].transform.position = GetPos();
            objects_red[i].transform.Rotate(0, Random.Range(0f, 90.0f), 0);
            objects_red[i].GetComponent<Rigidbody>().velocity = Vector3.zero;
        }

        this.objects_blue = new GameObject[this.num_resource_blue];
        for (int i = 0; i < this.num_resource_blue; i++)
        {
            GameObject obj = Instantiate(prefab_blue) as GameObject;
            objects_blue[i] = obj;
            objects_blue[i].transform.position = GetPos();
            objects_blue[i].transform.Rotate(0, Random.Range(0f, 90.0f), 0);
            objects_blue[i].GetComponent<Rigidbody>().velocity = Vector3.zero;
        }

    }

    public Vector3 GetPos()
    {
        return new Vector3(Random.Range(-this.objects_initial_range, this.objects_initial_range),
                        this.objects_initial_height,
                        Random.Range(-this.objects_initial_range, this.objects_initial_range));
    }

    public Vector3 GetPos(float offset_y)
    {
        return new Vector3(Random.Range(-this.objects_initial_range, this.objects_initial_range),
                        this.objects_initial_height + offset_y,
                        Random.Range(-this.objects_initial_range, this.objects_initial_range));
    }

    public override void OnEpisodeBegin()
    {
        print("New episode begin");

        for (int i = 0; i < this.NUM_RESOURCES; i++)
        {
            this.resource_levels[i] = 0;
        }
        this.taking_eat_behavior = false;

        // Randomize agent position
        this.transform.position = new Vector3(0, 1.5f, 0);
        this.transform.Rotate(0, Random.Range(0f, 360.0f), 0);

        // Randomize object positions
        for (int i = 0; i < this.num_resource_red; i++)
        {
            Vector3 pos = GetPos();
            objects_red[i].transform.position = pos;
            objects_red[i].transform.Rotate(0, Random.Range(0f, 90.0f), 0);
            objects_red[i].GetComponent<Rigidbody>().velocity = Vector3.zero;
        }
        for (int i = 0; i < this.num_resource_blue; i++)
        {
            Vector3 pos = GetPos();
            objects_blue[i].transform.position = pos;
            objects_blue[i].transform.Rotate(0, Random.Range(0f, 90.0f), 0);
            objects_blue[i].GetComponent<Rigidbody>().velocity = Vector3.zero;
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(resource_levels);
    }


    public void Update()
    {
        if (this.StepCount % 5 == 0)
        {
            RequestDecision();
        }
        else
        {
            RequestAction();
        }

        if (!this.character.isGrounded)
        {
            this.vertial_velocity -= 9.81f * Time.fixedDeltaTime;
            this.character.Move(new Vector3(0, -9.81f * Time.fixedDeltaTime));
        }
        else
        {
            this.vertial_velocity = 0;
        }

        for (int i = 0; i < this.NUM_RESOURCES; i++)
        {
            this.resource_levels[i] -= this.resource_decreasing_rate * Time.fixedDeltaTime;
        }
        //print(this.resource_decreasing_rate * Time.deltaTime);  


        //print("Resource (red, blue) = (" + this.resource_levels[0] + "," + this.resource_levels[1] + ")");
        Monitor.Log("Red", this.resource_levels[0] / 15f, transform);
        resource_red = this.resource_levels[0];
        Monitor.Log("Blue", this.resource_levels[1] / 15f, transform);
        resource_blue = this.resource_levels[1];

        Physics.Simulate(Time.fixedDeltaTime);
        Application.targetFrameRate = (int)(60.0 * Time.timeScale);
    }


    public override void OnActionReceived(float[] vectorAction)
    {
        // Get the action index for movement
        int action = Mathf.FloorToInt(vectorAction[0]);
        /*** Action Category
         * 0 : None 
         * 1 : Forward  
         * 2 : Left
         * 3 : Right
         * 4 : Eat
         * ***/

        this.taking_eat_behavior = false;
        switch (action)
        {
            case 0:
                break;
            case 1:
                Vector3 forward = transform.transform.forward;
                this.character.Move(this.forward_speed * forward * Time.deltaTime);
                break;
            case 2:
                this.transform.Rotate(0, -this.rotation_angle_speed * Time.deltaTime, 0);
                break;
            case 3:
                this.transform.Rotate(0, this.rotation_angle_speed * Time.deltaTime, 0);
                break;
            case 4:
                print("Eat Behavior");
                this.taking_eat_behavior = true;
                break;
        }

        // Set zero reward (reward will be defined in the learning-side)
        SetReward(0.0f);
        float max_deviation = Mathf.Max(Mathf.Abs(this.resource_levels[0]), Mathf.Abs(this.resource_levels[1]));
        if (max_deviation > this.resource_limit )
        {
            print("Agent died... (-_-;) ");
            EndEpisode();
        }
    }

    public override void Heuristic(float[] actionsOut)
    {
        if (Input.GetKey(KeyCode.UpArrow))
        {
            actionsOut[0] = 1f;
            print("forward");
        }
        else if (Input.GetKey(KeyCode.LeftArrow))
        {
            actionsOut[0] = 2f;
            print("left");
        }
        else if (Input.GetKey(KeyCode.RightArrow))
        {
            actionsOut[0] = 3f;
            print("right");
        }
        else if (Input.GetKey(KeyCode.Space))
        {
            actionsOut[0] = 4f;
            print("eat");
        }
        else
        {
            actionsOut[0] = 0f;
        }
    }

    public void IncreaseResource(string tag)
    {
        if (tag == "food_red")
        {
            this.resource_levels[0] += this.food_red_increase;
        }
        if (tag == "food_blue")
        {
            this.resource_levels[1] += this.food_blue_increase;
        }
    }

    public bool IsAgentTakingEatBehavior()
    {
        return this.taking_eat_behavior;
    }
}
