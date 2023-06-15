package com.ces.hospitalcare.controllers;
import com.ces.hospitalcare.http.request.AuthenticationRequest;
import com.ces.hospitalcare.http.request.RegisterRequest;
import com.ces.hospitalcare.http.response.RegisterResponse;
import com.ces.hospitalcare.http.response.UserResponse;
import com.ces.hospitalcare.service.IAuthenticationService;
import jakarta.mail.MessagingException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/guest/auth")
public class AuthenticationController {
  @Autowired
  private IAuthenticationService authenticationService;

  @PostMapping("/register")
  public ResponseEntity<RegisterResponse> register(
      @RequestBody RegisterRequest request
  ) throws MessagingException, UnsupportedEncodingException {
    RegisterResponse registeredUser = authenticationService.register(request);
    return ResponseEntity.status(HttpStatus.CREATED).body(registeredUser);
  }

  @PostMapping("/verify")
  public ResponseEntity<RegisterResponse> verifyEmailToken(
      @RequestParam("token") String verifyToken
  ) {
    RegisterResponse registeredUser = authenticationService.verifyToken(verifyToken);
    return ResponseEntity.status(HttpStatus.CREATED).body(registeredUser);
  }

  @PostMapping("/authenticate")
  public ResponseEntity<UserResponse> authenticate(
      @RequestBody AuthenticationRequest requestAuth,
      HttpServletResponse response
  ) {
    UserResponse userResponse = authenticationService.authenticate(requestAuth, response);
    return ResponseEntity.status(HttpStatus.ACCEPTED).body(userResponse);
  }

  @PostMapping("/refresh-token")
  public void refreshToken(
      HttpServletRequest request,
      HttpServletResponse response
  ) throws IOException {
    authenticationService.refreshToken(request, response);
  }
}