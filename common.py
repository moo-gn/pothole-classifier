def convert_milliseconds_to_timestamp(millis: float) -> str:
    seconds=(millis/1000)%60
    seconds = int(seconds)
    minutes=(millis/(1000*60))%60
    minutes = int(minutes)
    hours=(millis/(1000*60*60))%24
    hours = int(hours)
    seconds, minutes, hours = str(seconds).zfill(2), str(minutes).zfill(2), str(hours).zfill(2)
    return "%s:%s:%s" % (hours, minutes, seconds)