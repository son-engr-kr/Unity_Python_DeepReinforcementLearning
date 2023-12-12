using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;

public class UIManager : MonoBehaviour
{
    Label LabelSuccessRate;
    Label LabelRewardSum;
    float RewardSum = 0;
    Label LabelIterationCount;
    int TimeLimitMillis = 20 * 1000;
    // Start is called before the first frame update
    void Start()
    {
        var root = GetComponent<UIDocument>().rootVisualElement;
        LabelSuccessRate = root.Q<Label>("LabelSuccessRate");
        LabelRewardSum = root.Q<Label>("LabelRewardSum");
        LabelIterationCount = root.Q<Label>("LabelIterationCount");
    }

    // Update is called once per frame
    void Update()
    {
        LabelRewardSum.text = $"{RewardSum:n5}";
        //if (SuccessCount + FailCount == 0)
        //{
        //    LabelSuccessRate.text = "---%";
        //}
        //else
        //{
        //    LabelSuccessRate.text = $"{SuccessCount / (float)(SuccessCount + FailCount) * 100f:n1}%";
        //}
        //LabelIterationCount.text = $"{SuccessCount + FailCount}";
    }
}
