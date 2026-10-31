package com.clonephy.nxcontrol;

import android.app.Service;
import android.content.Intent;
import android.os.IBinder;
import android.provider.Settings;

public class BubbleService extends Service {
    @Override public int onStartCommand(Intent intent,int flags,int startId){
        Intent access = new Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS);
        access.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        startActivity(access);
        stopSelf();
        return START_NOT_STICKY;
    }
    @Override public IBinder onBind(Intent intent){return null;}
}
