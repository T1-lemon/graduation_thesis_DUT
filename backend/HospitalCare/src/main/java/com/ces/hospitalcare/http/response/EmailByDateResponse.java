package com.ces.hospitalcare.http.response;
import java.util.Date;

public interface EmailByDateResponse {
  Date getStartTime();
  String getEmail();
  String getFirstNameDoctor();
  String getLastNameDoctor();
  String getFirstNamePatient();
  String getLastNamePatient();
}
