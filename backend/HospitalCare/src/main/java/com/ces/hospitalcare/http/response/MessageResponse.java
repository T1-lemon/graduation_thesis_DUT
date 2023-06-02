package com.ces.hospitalcare.http.response;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class MessageResponse {
  private Boolean fromSelf;

  private String message;
}
