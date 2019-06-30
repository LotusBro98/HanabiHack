package com.example.hanabihack;

import android.content.Intent;
import android.os.AsyncTask;
import android.os.Bundle;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import java.io.InputStream;
import java.io.StringWriter;
import java.net.HttpURLConnection;
import java.net.URL;

public class AuthActivity extends AppCompatActivity {
    private WebView webView;

    private String auth_url = "https://github.com/login/oauth/authorize?client_id=Iv1.16b81f8ffae8b22e";
    private String token_url = "https://github.com/login/oauth/access_token?" +
            "client_id=Iv1.16b81f8ffae8b22e&" +
            "client_secret=bea31fa44d091f34b55e3be511c997e3a8c21d57&" +
            "code=";

    public class MyWebViewClient extends WebViewClient {
        @Override
        public boolean shouldOverrideUrlLoading(WebView view, String url) {
            if (url.startsWith("myurl://")) {
                String code = url.split("code=")[1];

                RequestTokenTask task = new RequestTokenTask();
                task.execute(code);
            } else {
                view.loadUrl(url);
            }
            return true;
        }
    }

    class RequestTokenTask extends AsyncTask<String, Void, String> {

        @Override
        protected void onPreExecute() {
            super.onPreExecute();
        }

        @Override
        protected String doInBackground(String... codes) {
            try {
                String code = codes[0];
                String newurl = token_url + code;

                URL url_send = new URL(newurl);
                HttpURLConnection conn = (HttpURLConnection) url_send.openConnection();
                conn.setRequestMethod("POST");
                conn.setDoOutput(true);

                InputStream rd = conn.getInputStream();
                byte[] token_bytes = new byte[1024];
                rd.read(token_bytes);
                String token_string = new String(token_bytes);
                token_string = token_string.split("access_token=")[1].split("&scope=")[0];

                System.out.println(token_string);
                return token_string;
            } catch (Exception e) {
                e.printStackTrace();
                return null;
            }
        }

        @Override
        protected void onPostExecute(String token) {
            System.out.println(token);
            Intent i = new Intent(AuthActivity.this, ProfileActivity.class);
            startActivity(i);
        }
    }

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.auth_view);
        webView = findViewById(R.id.web_auth);
        webView.setWebViewClient(new MyWebViewClient());
//        webView.getSettings().setJavaScriptEnabled(true);
        webView.loadUrl(auth_url);
    }



}
