package com.example.hanabihack;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.webkit.WebView;
import android.widget.Button;
import android.widget.EditText;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;

public class LoginActivity extends AppCompatActivity {

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.login_page);

        Button bLogin = findViewById(R.id.bLogin);
        bLogin.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent i = new Intent(LoginActivity.this, AuthActivity.class);
                startActivity(i);

//                Intent i = new Intent(Intent.ACTION_VIEW);
//                i.setData(Uri.parse(auth_url));
//                startActivity(i);
            }
        });
    }

    public String requestGitHubToken(String email, String password)
    {
        try {


            return null;

        } catch (Exception e) {
            //textView.setText(e.toString());
        }
        return null;
    }
}
