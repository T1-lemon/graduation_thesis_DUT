package com.ces.hospitalcare.service;
import com.ces.hospitalcare.http.request.AuthenticationRequest;
import com.ces.hospitalcare.http.request.RegisterRequest;
import com.ces.hospitalcare.http.response.RegisterResponse;
import com.ces.hospitalcare.http.response.UserResponse;
import jakarta.mail.MessagingException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.UnsupportedEncodingException;

public interface IAuthenticationService {
  RegisterResponse register(RegisterRequest request)
      throws MessagingException, UnsupportedEncodingException;

  UserResponse authenticate(AuthenticationRequest request, HttpServletResponse response);

  void refreshToken(HttpServletRequest request, HttpServletResponse response);
  public RegisterResponse verifyToken(String verifyToken);
}
