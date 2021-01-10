using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FoodEating : MonoBehaviour
{
    public ForagerAgent agent;

    // Start is called before the first frame update
    void Start()
    {

    }

    public void OnTriggerStay(Collider other)
    {
        bool is_eaten = false;
        if (agent.IsAgentTakingEatBehavior())
        {
            if (other.CompareTag("food_red"))
            {
                agent.IncreaseResource("food_red");
                is_eaten = true;
            }
            else if (other.CompareTag("food_blue"))
            {
                agent.IncreaseResource("food_blue");
                is_eaten = true;
            }
        }

        if (is_eaten)
        {
            other.transform.position = agent.GetPos();
        }
    }
}
