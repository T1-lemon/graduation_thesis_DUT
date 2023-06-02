package com.ces.hospitalcare.util;
import com.ces.hospitalcare.security.config.SecurityContact;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.http.HttpHeaders;
import org.springframework.http.ResponseCookie;
import org.springframework.stereotype.Component;

@Component
public class CookieHandler {
  public void deleteTokenCookie(HttpServletResponse response) {
    ResponseCookie cookieForAccessToken = generateCookie("token", "/", "", 0);
    ResponseCookie cookieForRefreshToken = generateCookie("refreshToken",
        "/api/v1/guest/auth/refresh-token", "", 0);

    response.addHeader(HttpHeaders.SET_COOKIE, cookieForRefreshToken.toString());
    response.addHeader(HttpHeaders.SET_COOKIE, cookieForAccessToken.toString());
  }

  public ResponseCookie generateCookie(String cookieName, String path, String cookieValue,
      long expiration) {
    ResponseCookie cookie = ResponseCookie.from(cookieName, cookieValue)
        .httpOnly(true)
        .maxAge(expiration)
        .path(path)
//        .sameSite("None")
//        .secure(true)
        .build();
    return cookie;
  }

  public void setCookieToClient(HttpServletResponse response, String accessToken,
      String refreshToken) {
    ResponseCookie cookieForAccessToken = generateCookie("token", "/", accessToken,
        SecurityContact.EXPIRATION_TIME);
    ResponseCookie cookieForRefreshToken = generateCookie("refreshToken",
        "/api/v1/guest/auth/refresh-token", refreshToken, SecurityContact.REFRESH_EXPIRATION_TIME);

    response.addHeader(HttpHeaders.SET_COOKIE, cookieForAccessToken.toString());
    response.addHeader(HttpHeaders.SET_COOKIE, cookieForRefreshToken.toString());
  }
}