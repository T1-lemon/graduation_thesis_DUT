package com.ces.hospitalcare.util;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;

public class PepperedBCryptPasswordEncoder implements PasswordEncoder {
  private final String pepper;

  private final BCryptPasswordEncoder delegate;

  public PepperedBCryptPasswordEncoder(String pepper) {
    this.pepper = pepper;
    this.delegate = new BCryptPasswordEncoder();
  }

  //Method to add pepper before hash rawPassword
  @Override
  public String encode(CharSequence rawPassword) {
    String pepperPassword = rawPassword + pepper;
    return delegate.encode(pepperPassword);
  }

  //Method to check match rawPassword from User and encodedPassword from database
  @Override
  public boolean matches(CharSequence rawPassword, String encodedPassword) {
    String pepperPassword = rawPassword + pepper;
    return delegate.matches(pepperPassword, encodedPassword);
  }
}