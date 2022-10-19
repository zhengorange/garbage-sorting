package com.example.sort.ResponseTemplate;

import lombok.Data;
import lombok.experimental.Accessors;

@Data
@Accessors(chain = true)
public class ResBody {
    Integer code = ErrorCode.OK;
    String msg = "操作完成";
    Object data;
}
