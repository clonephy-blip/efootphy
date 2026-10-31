package com.clonephy.nxcontrol;

import android.app.Activity;
import android.graphics.Color;
import android.os.Bundle;
import android.view.ViewGroup;
import android.widget.*;
import java.util.concurrent.*;

public class TerminalActivity extends Activity {
    final ExecutorService io=Executors.newSingleThreadExecutor();
    TextView out;
    EditText cmd;
    @Override public void onCreate(Bundle b){super.onCreate(b);build();}
    int dp(int n){return (int)(n*getResources().getDisplayMetrics().density+.5f);}
    void build(){
        LinearLayout root=new LinearLayout(this);root.setOrientation(LinearLayout.VERTICAL);root.setPadding(dp(16),dp(18),dp(16),dp(18));root.setBackgroundColor(Color.rgb(16,11,7));
        TextView title=new TextView(this);title.setText("Terminal ROOT");title.setTextSize(26);title.setTextColor(Color.rgb(249,226,207));title.setTypeface(null,1);root.addView(title);
        cmd=new EditText(this);cmd.setHint("Nhập lệnh shell...");cmd.setTextColor(Color.WHITE);cmd.setHintTextColor(Color.GRAY);cmd.setSingleLine(false);root.addView(cmd,new LinearLayout.LayoutParams(-1,dp(120)));
        Button run=new Button(this);run.setText("Chạy ROOT");run.setAllCaps(false);root.addView(run,new LinearLayout.LayoutParams(-1,dp(56)));
        ScrollView sv=new ScrollView(this);out=new TextView(this);out.setTextColor(Color.rgb(220,220,220));out.setTextSize(13);out.setTextIsSelectable(true);out.setPadding(0,dp(12),0,0);sv.addView(out);root.addView(sv,new LinearLayout.LayoutParams(-1,0,1));setContentView(root);
        run.setOnClickListener(v->{String c=cmd.getText().toString();out.setText("$ "+c+"\n...");io.execute(()->{String r=RootBridge.run(c);runOnUiThread(()->out.setText("$ "+c+"\n"+r));});});
    }
    @Override protected void onDestroy(){super.onDestroy();io.shutdownNow();}
}
