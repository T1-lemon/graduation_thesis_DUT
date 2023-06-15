package com.ces.hospitalcare.http.response;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class CheckResponse {
  private Integer statusCode;

  private String message;

  private boolean isBooked;
}
