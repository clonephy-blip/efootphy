package com.clonephy.nxcontrol;

import android.app.Activity;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Base64;
import android.widget.*;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.*;

public class ScriptActivity extends Activity {
    final ExecutorService io=Executors.newSingleThreadExecutor();
    EditText name,body; TextView out,list;
    @Override public void onCreate(Bundle b){super.onCreate(b);build();refreshList();}
    int dp(int n){return (int)(n*getResources().getDisplayMetrics().density+.5f);}
    TextView tv(String s,int sp){TextView t=new TextView(this);t.setText(s);t.setTextSize(sp);t.setTextColor(Color.rgb(249,226,207));return t;}
    void build(){
        LinearLayout root=new LinearLayout(this);root.setOrientation(LinearLayout.VERTICAL);root.setPadding(dp(16),dp(18),dp(16),dp(18));root.setBackgroundColor(Color.rgb(16,11,7));TextView title=tv("Kho Script",26);title.setTypeface(null,1);root.addView(title);root.addView(tv("/sdcard/NXControl/scripts",12));
        list=tv("",12);root.addView(list,new LinearLayout.LayoutParams(-1,dp(80)));
        name=new EditText(this);name.setHint("Tên file, ví dụ boost.sh");name.setText("script.sh");name.setTextColor(Color.WHITE);name.setHintTextColor(Color.GRAY);root.addView(name);
        body=new EditText(this);body.setHint("Nội dung shell script");body.setText("#!/system/bin/sh\n");body.setTextColor(Color.WHITE);body.setHintTextColor(Color.GRAY);body.setGravity(android.view.Gravity.TOP);root.addView(body,new LinearLayout.LayoutParams(-1,0,1));
        LinearLayout row=new LinearLayout(this);Button save=new Button(this),run=new Button(this),refresh=new Button(this);save.setText("Lưu");run.setText("Chạy ROOT");refresh.setText("Làm mới");row.addView(save,new LinearLayout.LayoutParams(0,dp(56),1));row.addView(run,new LinearLayout.LayoutParams(0,dp(56),1));row.addView(refresh,new LinearLayout.LayoutParams(0,dp(56),1));root.addView(row);out=tv("",12);ScrollView sv=new ScrollView(this);sv.addView(out);root.addView(sv,new LinearLayout.LayoutParams(-1,dp(180)));setContentView(root);
        save.setOnClickListener(v->save());run.setOnClickListener(v->runScript());refresh.setOnClickListener(v->refreshList());
    }
    String safe(){return name.getText().toString().replaceAll("[^A-Za-z0-9_.-]","_");}
    void async(String cmd,java.util.function.Consumer<String> cb){io.execute(()->{String r=RootBridge.run(cmd);runOnUiThread(()->cb.accept(r));});}
    void save(){String b64=Base64.encodeToString(body.getText().toString().getBytes(StandardCharsets.UTF_8),Base64.NO_WRAP);async(NxQueries.saveScript(safe(),b64),r->{out.setText(r);refreshList();});}
    void runScript(){async(NxQueries.runScript(safe()),out::setText);}
    void refreshList(){async(NxQueries.listScripts(),r->list.setText("Đã lưu:\n"+(r.isEmpty()?"(chưa có)":r)));}
    @Override protected void onDestroy(){super.onDestroy();io.shutdownNow();}
}
