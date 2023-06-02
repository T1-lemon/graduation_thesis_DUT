package com.ces.hospitalcare.service.impl;
import com.ces.hospitalcare.builder.UserBuilder;
import com.ces.hospitalcare.dto.UserDTO;
import com.ces.hospitalcare.entity.UserEntity;
import com.ces.hospitalcare.http.exception.AlreadyExistException;
import com.ces.hospitalcare.http.exception.LoginUserException;
import com.ces.hospitalcare.http.exception.ResourceNotFoundException;
import com.ces.hospitalcare.http.request.AuthenticationRequest;
import com.ces.hospitalcare.http.request.RegisterRequest;
import com.ces.hospitalcare.http.response.RegisterResponse;
import com.ces.hospitalcare.http.response.UserResponse;
import com.ces.hospitalcare.repository.UserRepository;
import com.ces.hospitalcare.security.service.JwtService;
import com.ces.hospitalcare.security.service.ValidTokenService;
import com.ces.hospitalcare.service.IAuthenticationService;
import com.ces.hospitalcare.util.CookieHandler;
import com.ces.hospitalcare.util.ExceptionMessage;
import jakarta.servlet.http.Cookie;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import org.modelmapper.ModelMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.web.util.WebUtils;

@Service
@RequiredArgsConstructor
public class AuthenticationServiceImpl implements IAuthenticationService {
  @Autowired
  private UserRepository userRepository;

  @Autowired
  private PasswordEncoder passwordEncoder;

  @Autowired
  private JwtService jwtService;

  @Autowired
  private UserBuilder userBuilder;

  @Autowired
  private ModelMapper mapper;

  @Autowired
  private ValidTokenService validTokenService;

  @Autowired
  private CookieHandler cookieHandler;

  public RegisterResponse register(RegisterRequest request) {
    if (userRepository.findByEmail(request.getEmail()).isEmpty() == false) {
      throw new AlreadyExistException("Email already exists");
    }

    UserEntity user = userBuilder.userEntityBuild(request,
        passwordEncoder.encode(request.getPassword()));
    userRepository.save(user);

    return RegisterResponse.builder()
        .message("Register successfully")
        .status(HttpStatus.CREATED.value())
        .build();
  }

  public UserResponse authenticate(AuthenticationRequest request, HttpServletResponse response) {
    var user = userRepository.findByEmail(request.getEmail())
        .orElseThrow(() -> new LoginUserException(
            ExceptionMessage.USERNAME_PASSWORD_INVALIDATE.getMessage()));

    boolean validatePassword = passwordEncoder.matches(request.getPassword(), user.getPassword());
    if (!validatePassword) {
      throw new LoginUserException(ExceptionMessage.USERNAME_PASSWORD_INVALIDATE.getMessage());
    }

    String accessToken = validTokenService.getAccessToken(user.getEmail());
    String refreshToken = validTokenService.getRefreshToken(user.getEmail());

    if (accessToken == null) {
      accessToken = jwtService.generateToken(user);
      validTokenService.setAccessToken(accessToken);
    }

    if (refreshToken == null) {
      refreshToken = jwtService.generateRefreshToken(user);
      validTokenService.setRefreshToken(refreshToken);
    }

    cookieHandler.setCookieToClient(response, accessToken, refreshToken);
    UserDTO userDTO = mapper.map(user, UserDTO.class);
    return UserResponse.builder()
        .user(userDTO)
        .status(HttpStatus.ACCEPTED.value())
        .build();
  }

  @Override
  public void refreshToken(HttpServletRequest request, HttpServletResponse response) {
    String refreshToken;
    String userEmail;

    Cookie cookie = WebUtils.getCookie(request, "refreshToken");
    try {
      refreshToken = cookie.getValue();
      if (!validTokenService.isRefreshTokenExists(refreshToken)) {
        throw new Exception();
      }
      userEmail = jwtService.extractUsername(refreshToken);
    } catch (Exception e) {
      cookieHandler.deleteTokenCookie(response);
      throw new ResourceNotFoundException(ExceptionMessage.INVALID_REFRESH_TOKEN.getMessage());
    }

    UserEntity user = userRepository.findByEmail(userEmail)
        .orElseThrow();

    String accessToken = jwtService.generateToken(user);
    validTokenService.setAccessToken(accessToken);

    refreshToken = jwtService.generateRefreshToken(user);
    validTokenService.setRefreshToken(refreshToken);

    cookieHandler.setCookieToClient(response, accessToken, refreshToken);
  }
}
