package com.ces.hospitalcare.controllers;
import com.ces.hospitalcare.dto.NotificationDTO;
import com.ces.hospitalcare.dto.UserDTO;
import com.ces.hospitalcare.http.request.NotificationRequest;
import com.ces.hospitalcare.http.response.NotificationPatientResponse;
import com.ces.hospitalcare.http.response.UserResponse;
import com.ces.hospitalcare.service.INotificationService;
import com.ces.hospitalcare.service.IUserService;
import java.util.List;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/notifications/")
public class NotificationController {
  @Autowired
  private INotificationService notificationService;

  @Autowired
  private IUserService userService;

  @GetMapping(path = "doctor")
  @PreAuthorize("hasAnyAuthority( 'ROLE_DOCTOR')")
  public List<NotificationDTO> getNotificationsDoctor() {
    UserResponse userResponse = userService.getCurrentUser();
    UserDTO doctor = userResponse.getUser();
    return notificationService.getNotificationsDoctor(doctor.getId());
  }

  @GetMapping(path = "patient")
  @PreAuthorize("hasAnyAuthority( 'ROLE_PATIENT')")
  public List<NotificationPatientResponse> getNotificationsPatient() {
    UserResponse userResponse = userService.getCurrentUser();
    UserDTO patient = userResponse.getUser();
    return notificationService.getNotificationsPatient(patient.getId());
  }

  @PatchMapping(path = "read-notification")
  @PreAuthorize("hasAnyAuthority( 'ROLE_PATIENT','ROLE_DOCTOR')")
  public NotificationDTO readNotifications(@RequestBody NotificationRequest notificationRequest) {
    UserResponse userResponse = userService.getCurrentUser();
    UserDTO userLogin = userResponse.getUser();
    notificationRequest.setUserId(userLogin.getId());
    return notificationService.changeReadNotification(notificationRequest);
  }
}
