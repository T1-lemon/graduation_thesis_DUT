package com.ces.hospitalcare.http.exception;
public class VerifyTokenException extends RuntimeException{
  public VerifyTokenException() {
    super();
  }

  public VerifyTokenException(String message) {
    super(message);
  }

  public VerifyTokenException(String message, Throwable cause) {
    super(message, cause);
  }

  public VerifyTokenException(Throwable cause) {
    super(cause);
  }

  protected VerifyTokenException(String message, Throwable cause, boolean enableSuppression,
      boolean writableStackTrace) {
    super(message, cause, enableSuppression, writableStackTrace);
  }
}
