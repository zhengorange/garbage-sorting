package com.example.sort.controller;
import com.example.sort.util.RPCClient;
import com.example.sort.ResponseTemplate.ResBody;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;


@Slf4j
@RestController
@RequestMapping("image")
public class MainController {
    private final RPCClient rpcClient;

    @Autowired
    MainController(RPCClient rpcClient){
        this.rpcClient = rpcClient;
    }

    @PostMapping
    public ResBody image(@RequestBody ImageData imageData) throws IOException, InterruptedException {
        return new ResBody().setData(rpcClient.call(imageData.getImage()));
    }
}
