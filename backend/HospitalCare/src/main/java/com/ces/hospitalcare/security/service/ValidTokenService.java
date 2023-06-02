package com.ces.hospitalcare.security.service;
import com.ces.hospitalcare.util.CookieHandler;
import jakarta.servlet.http.Cookie;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.util.HashMap;
import java.util.Map;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.Authentication;
import org.springframework.security.web.authentication.logout.LogoutHandler;
import org.springframework.stereotype.Service;
import org.springframework.web.util.WebUtils;

@Service
@Data
@NoArgsConstructor
@AllArgsConstructor
public class ValidTokenService implements LogoutHandler {
  @Autowired
  JwtService jwtService;

  @Autowired
  CookieHandler cookieHandler;

  private Map<String, String> accessToken = new HashMap<>();

  private Map<String, String> refreshToken = new HashMap<>();

  public void setAccessToken(String token) {
    String userEmail = jwtService.extractUsername(token);
    accessToken.put(userEmail, token);
  }

  public String getAccessToken(String userEmail) {
    return accessToken.get(userEmail);
  }

  public void setRefreshToken(String token) {
    String userEmail = jwtService.extractUsername(token);
    refreshToken.put(userEmail, token);
  }

  public String getRefreshToken(String userEmail) {
    return refreshToken.get(userEmail);
  }

  public boolean isRefreshTokenExists(String token) {
    return token.equals(refreshToken.get(jwtService.extractUsername(token)));
  }

  public boolean isAccessTokenExists(String token) {
    return token.equals(accessToken.get(jwtService.extractUsername(token)));
  }

  @Override

  public void logout(
      HttpServletRequest request,
      HttpServletResponse response,
      Authentication authentication) {

    Cookie token = WebUtils.getCookie(request, "token");
    if (token == null) {
      return;
    }

    String userEmail = jwtService.extractUsername(token.getValue());

    accessToken.remove(userEmail);
    refreshToken.remove(userEmail);
    cookieHandler.deleteTokenCookie(response);
  }
}
