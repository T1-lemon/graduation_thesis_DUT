package com.ces.hospitalcare.security.filter;
import com.ces.hospitalcare.http.response.ErrorResponse;
import com.ces.hospitalcare.security.service.JwtService;
import com.ces.hospitalcare.security.service.ValidTokenService;
import com.ces.hospitalcare.util.ExceptionMessage;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.Cookie;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.Date;
import lombok.NonNull;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.web.authentication.WebAuthenticationDetailsSource;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;
import org.springframework.web.util.WebUtils;

@Component
@RequiredArgsConstructor
public class JwtAuthenticationFilter extends OncePerRequestFilter {
  private final JwtService jwtService;

  private final UserDetailsService userDetailsService;

  @Autowired
  private ValidTokenService validTokenService;

  @Override
  protected void doFilterInternal(
      @NonNull HttpServletRequest request,
      @NonNull HttpServletResponse response,
      @NonNull FilterChain filterChain
  ) throws ServletException, IOException {
    if (request.getRequestURI().startsWith("/api/v1/guest/")) {
      filterChain.doFilter(request, response);
      return;
    }

    final String accessToken;
    final String userEmail;
    Cookie token = WebUtils.getCookie(request, "token");

    try {
      accessToken = token.getValue();
      if (!validTokenService.isAccessTokenExists(accessToken)) {
        throw new Exception();
      }
      userEmail = jwtService.extractUsername(accessToken);
    } catch (Exception e) {
      ErrorResponse errorResponse = new ErrorResponse(
          ExceptionMessage.INVALID_ACCESS_TOKEN.getMessage(), new Date(),
          HttpStatus.FORBIDDEN.value());
      response.setContentType("application/json");
      response.setStatus(HttpStatus.FORBIDDEN.value());
      response.getWriter().write(new ObjectMapper().writeValueAsString(errorResponse));
      return;
    }

    UserDetails userDetails = this.userDetailsService.loadUserByUsername(userEmail);
    UsernamePasswordAuthenticationToken authToken = new UsernamePasswordAuthenticationToken(
        userDetails,
        null,
        userDetails.getAuthorities()
    );
    authToken.setDetails(
        new WebAuthenticationDetailsSource().buildDetails(request)
    );
    SecurityContextHolder.getContext().setAuthentication(authToken);
    filterChain.doFilter(request, response);
  }
}
