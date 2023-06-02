package com.ces.hospitalcare.service;
import com.ces.hospitalcare.http.request.AuthenticationRequest;
import com.ces.hospitalcare.http.request.RegisterRequest;
import com.ces.hospitalcare.http.response.RegisterResponse;
import com.ces.hospitalcare.http.response.UserResponse;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

public interface IAuthenticationService {
  RegisterResponse register(RegisterRequest request);

  UserResponse authenticate(AuthenticationRequest request, HttpServletResponse response);

  void refreshToken(HttpServletRequest request, HttpServletResponse response);
}
