package com.clonephy.nxcontrol;

import android.app.Service;
import android.content.Intent;
import android.os.IBinder;

public class BubbleService extends Service {
    @Override public int onStartCommand(Intent intent,int flags,int startId){
        Intent launch=getPackageManager().getLaunchIntentForPackage("com.clonephy.nxcharge");
        if(launch!=null){launch.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);startActivity(launch);}
        stopSelf();
        return START_NOT_STICKY;
    }
    @Override public IBinder onBind(Intent intent){return null;}
}
