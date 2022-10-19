package com.example.sort.ResponseTemplate;

public class ErrorCode {
    public final static Integer OK = 1000;           // all right
    public final static Integer TOKEN_EXPIRED =2000; // token过期
    public final static Integer TOKEN_ERROR =2100;   // token错误
    public final static Integer WORK_ERROR = 3000;   // 业务异常
    public final static Integer SERVER_ERROR = 5000; // 系统错误

}
