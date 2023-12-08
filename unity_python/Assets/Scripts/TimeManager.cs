using UnityEngine.SocialPlatforms;
using UnityEngine;

public class TimeManager
{
    public static long GetSimulationTimeMillis()
    {
        return (long)(Time.time * 1000);

    }
    public static long GetRealNowTimeMillis()
    {
        return System.DateTimeOffset.Now.ToUnixTimeMilliseconds();
    }
}