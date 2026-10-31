package com.clonephy.nxcontrol;

import android.app.*;
import android.content.*;
import android.content.pm.*;
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.provider.Settings;
import android.view.Gravity;
import android.view.View;
import android.widget.*;
import java.util.*;
import java.util.concurrent.*;

public class MainActivity extends Activity {
    static final int BG=Color.rgb(16,11,7), CARD=Color.rgb(42,27,17), COPPER=Color.rgb(139,84,39), TEXT=Color.rgb(249,226,207), MUTED=Color.rgb(188,158,137);
    final ExecutorService io=Executors.newCachedThreadPool();
    SharedPreferences prefs;
    LinearLayout root,gamesBox;
    TextView cpuT,gpuT,ramT,bypassT,status;

    @Override public void onCreate(Bundle b){super.onCreate(b);prefs=getSharedPreferences("nx",MODE_PRIVATE);buildUi();refreshStats();}
    int dp(int n){return (int)(n*getResources().getDisplayMetrics().density+.5f);}    
    TextView tv(String s,int sp,int c){TextView v=new TextView(this);v.setText(s);v.setTextSize(sp);v.setTextColor(c);v.setPadding(dp(6),dp(5),dp(6),dp(5));return v;}
    GradientDrawable bg(int c,int r){GradientDrawable g=new GradientDrawable();g.setColor(c);g.setCornerRadius(dp(r));return g;}
    Button btn(String s){Button b=new Button(this);b.setText(s);b.setAllCaps(false);b.setTextColor(TEXT);b.setBackground(bg(COPPER,16));return b;}
    LinearLayout card(){LinearLayout c=new LinearLayout(this);c.setOrientation(LinearLayout.VERTICAL);c.setPadding(dp(16),dp(14),dp(16),dp(14));c.setBackground(bg(CARD,20));LinearLayout.LayoutParams p=new LinearLayout.LayoutParams(-1,-2);p.setMargins(dp(14),dp(7),dp(14),dp(7));c.setLayoutParams(p);return c;}
    TextView section(String s){TextView t=tv(s,19,TEXT);t.setTypeface(null,1);return t;}
    TextView metric(String n){TextView t=tv(n+"\n--",15,TEXT);t.setGravity(Gravity.CENTER);t.setBackground(bg(Color.rgb(58,35,21),16));return t;}

    void buildUi(){
        ScrollView sv=new ScrollView(this);sv.setBackgroundColor(BG);root=new LinearLayout(this);root.setOrientation(LinearLayout.VERTICAL);root.setPadding(0,dp(16),0,dp(30));sv.addView(root);setContentView(sv);
        TextView title=tv("NX Control Center",28,TEXT);title.setTypeface(null,1);title.setPadding(dp(20),dp(8),dp(20),0);root.addView(title);root.addView(tv("NX659J • ROOT ENGINE • v0.2",12,MUTED));

        LinearLayout stats=card();stats.addView(section("Tăng Tốc Game"));stats.addView(tv("Thông số phần cứng",13,MUTED));LinearLayout r=new LinearLayout(this);r.setOrientation(LinearLayout.HORIZONTAL);cpuT=metric("CPU");gpuT=metric("GPU");ramT=metric("RAM");r.addView(cpuT,new LinearLayout.LayoutParams(0,dp(76),1));r.addView(gpuT,new LinearLayout.LayoutParams(0,dp(76),1));r.addView(ramT,new LinearLayout.LayoutParams(0,dp(76),1));stats.addView(r);root.addView(stats);

        LinearLayout quick=card();quick.addView(section("Tác vụ 1 chạm"));LinearLayout q1=new LinearLayout(this),q2=new LinearLayout(this);Button bypass=btn("🔌 Sạc nhánh"),clean=btn("🧹 Dọn nhẹ"),boost=btn("⚡ Boost 30s"),cool=btn("❄ Hạ nhiệt 60s");q1.addView(bypass,new LinearLayout.LayoutParams(0,dp(58),1));q1.addView(clean,new LinearLayout.LayoutParams(0,dp(58),1));q2.addView(boost,new LinearLayout.LayoutParams(0,dp(58),1));q2.addView(cool,new LinearLayout.LayoutParams(0,dp(58),1));quick.addView(q1);quick.addView(q2);bypassT=tv("Sạc nhánh: ...",13,MUTED);quick.addView(bypassT);root.addView(quick);
        bypass.setOnClickListener(v->toggleBypass());clean.setOnClickListener(v->clean());boost.setOnClickListener(v->action(NxActions.boost30(),"Boost 30 giây đã bật"));cool.setOnClickListener(v->action(NxActions.cool60(),"Cool mode 60 giây đã bật"));

        LinearLayout games=card();LinearLayout gh=new LinearLayout(this);gh.setGravity(Gravity.CENTER_VERTICAL);gh.addView(section("Game Space"),new LinearLayout.LayoutParams(0,-2,1));Button plus=btn("＋");gh.addView(plus,new LinearLayout.LayoutParams(dp(64),dp(48)));games.addView(gh);gamesBox=new LinearLayout(this);gamesBox.setOrientation(LinearLayout.VERTICAL);games.addView(gamesBox);root.addView(games);plus.setOnClickListener(v->pickApps());renderGames();

        LinearLayout tools=card();tools.addView(section("Công cụ ROOT"));Button terminal=btn("⌨ Terminal"),scripts=btn("📜 Kho Script"),display=btn("🖥 Display / Resolution"),bubble=btn("◉ Bubble"),reset=btn("↺ Panic Reset");for(Button b:new Button[]{terminal,scripts,display,bubble,reset}){LinearLayout.LayoutParams p=new LinearLayout.LayoutParams(-1,dp(56));p.setMargins(0,dp(4),0,dp(4));tools.addView(b,p);}root.addView(tools);
        terminal.setOnClickListener(v->startActivity(new Intent(this,TerminalActivity.class)));scripts.setOnClickListener(v->startActivity(new Intent(this,ScriptActivity.class)));display.setOnClickListener(v->showDisplay());bubble.setOnClickListener(v->enableBubble());reset.setOnClickListener(v->reset());
        status=tv("Sẵn sàng",12,MUTED);status.setPadding(dp(20),dp(8),dp(20),dp(8));root.addView(status);
    }

    void rootAsync(String cmd,java.util.function.Consumer<String> cb){status.setText("ROOT...");io.execute(()->{String s=RootBridge.run(cmd);runOnUiThread(()->{status.setText(s.isEmpty()?"Hoàn tất":s);cb.accept(s);});});}
    void action(String cmd,String msg){rootAsync(cmd,r->Toast.makeText(this,msg,Toast.LENGTH_SHORT).show());}
    int temp(String s){try{int v=Integer.parseInt(s.trim());while(v>200)v/=10;return v;}catch(Exception e){return 0;}}
    void refreshStats(){rootAsync(NxQueries.telemetry(),r->{try{String[] a=r.split("\\|");cpuT.setText("CPU\n"+temp(a[0])+"°C");gpuT.setText("GPU\n"+temp(a[1])+"°C");long t=Long.parseLong(a[2]),av=Long.parseLong(a[3]);ramT.setText("RAM\n"+((t-av)*100/t)+"%");bypassT.setText("Sạc nhánh: "+("4".equals(a[4].trim())?"BẬT":"TẮT"));}catch(Exception ignored){}});}
    void toggleBypass(){rootAsync(NxQueries.bypassState(),r->{boolean on="4".equals(r.trim());rootAsync(on?NxActions.bypassOff():NxActions.bypassOn(),x->refreshStats());});}
    void clean(){rootAsync(NxActions.clean(),r->{try{String[] a=r.split("\\s+");long mb=Math.max(0,(Long.parseLong(a[1])-Long.parseLong(a[0]))/1024);new AlertDialog.Builder(this).setTitle("Dọn nhẹ RAM").setMessage("Đã giải phóng khoảng "+mb+" MB app nền.\nKhông dùng drop_caches.").setPositiveButton("Đóng",null).show();}catch(Exception ignored){}});}

    Set<String> apps(){return new HashSet<>(prefs.getStringSet("apps",Collections.emptySet()));}
    void renderGames(){gamesBox.removeAllViews();Set<String> set=apps();if(set.isEmpty()){gamesBox.addView(tv("Nhấn + để thêm app. Bubble chỉ hiện trong app đã chọn.",13,MUTED));return;}PackageManager pm=getPackageManager();for(String pkg:set){String label=pkg;try{label=pm.getApplicationLabel(pm.getApplicationInfo(pkg,0)).toString();}catch(Exception ignored){}LinearLayout row=new LinearLayout(this);row.setGravity(Gravity.CENTER_VERTICAL);row.addView(tv(label+"\n"+pkg,13,TEXT),new LinearLayout.LayoutParams(0,dp(64),1));Button p=btn("⚙"),go=btn("▶");row.addView(p,new LinearLayout.LayoutParams(dp(58),dp(48)));row.addView(go,new LinearLayout.LayoutParams(dp(58),dp(48)));gamesBox.addView(row);String x=pkg;p.setOnClickListener(v->ProfileDialogs.edit(this,prefs,x));go.setOnClickListener(v->{Intent i=pm.getLaunchIntentForPackage(x);if(i!=null)startActivity(i);});}}
    void pickApps(){PackageManager pm=getPackageManager();Intent in=new Intent(Intent.ACTION_MAIN);in.addCategory(Intent.CATEGORY_LAUNCHER);List<ResolveInfo> rs=pm.queryIntentActivities(in,0);rs.sort(Comparator.comparing(a->a.loadLabel(pm).toString().toLowerCase()));String[] names=new String[rs.size()];boolean[] checks=new boolean[rs.size()];Set<String> set=apps();for(int i=0;i<rs.size();i++){names[i]=rs.get(i).loadLabel(pm)+" · "+rs.get(i).activityInfo.packageName;checks[i]=set.contains(rs.get(i).activityInfo.packageName);}new AlertDialog.Builder(this).setTitle("Thêm app vào Game Space").setMultiChoiceItems(names,checks,(d,w,c)->{}).setPositiveButton("Lưu",(d,w)->{Set<String> n=new HashSet<>();for(int i=0;i<checks.length;i++)if(checks[i])n.add(rs.get(i).activityInfo.packageName);prefs.edit().putStringSet("apps",n).apply();renderGames();}).setNegativeButton("Hủy",null).show();}
    void showDisplay(){rootAsync(NxQueries.displayInfo(),r->new AlertDialog.Builder(this).setTitle("Display hiện tại").setMessage(r+"\n\nProfile riêng: Game Space → ⚙").setPositiveButton("OK",null).show());}
    void enableBubble(){if(!Settings.canDrawOverlays(this)){startActivity(new Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, Uri.parse("package:"+getPackageName())));Toast.makeText(this,"Cho phép hiển thị trên ứng dụng khác rồi nhấn Bubble lại.",Toast.LENGTH_LONG).show();return;}startActivity(new Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS));startService(new Intent(this,BubbleService.class));Toast.makeText(this,"Bubble service đã bật. Hãy cấp Usage Access.",Toast.LENGTH_LONG).show();}
    void reset(){new AlertDialog.Builder(this).setTitle("Panic Reset").setMessage("Reset display/DPI/Hz, CPU/GPU và tắt bypass?").setPositiveButton("RESET",(d,w)->action(NxActions.panicReset(),"Đã reset NX")).setNegativeButton("Hủy",null).show();}
    @Override protected void onDestroy(){super.onDestroy();io.shutdownNow();}
}
