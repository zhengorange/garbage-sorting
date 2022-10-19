package com.example.sort.exception;
import com.example.sort.ResponseTemplate.ErrorCode;
import com.example.sort.ResponseTemplate.ResBody;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseBody;

@Slf4j
@ControllerAdvice
public class GlobalExceptionHandlerRest {
    @ExceptionHandler(value = RuntimeException.class)
    @ResponseBody
    public ResBody controllerHandler(RuntimeException e){
        log.error(e.toString());
        return new ResBody().setCode(ErrorCode.WORK_ERROR).setMsg(e.getMessage());
    }
}
