package com.ces.hospitalcare.controllers;
import com.ces.hospitalcare.dto.TimeSlotDTO;
import com.ces.hospitalcare.dto.UserDTO;
import com.ces.hospitalcare.http.request.TimeSlotRequest;
import com.ces.hospitalcare.http.response.TimeSlotResponse;
import com.ces.hospitalcare.http.response.UserResponse;
import com.ces.hospitalcare.service.ITimeSlotService;
import com.ces.hospitalcare.service.IUserService;
import java.util.List;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping(path = "/api/v1")
public class TimeSlotController {
  @Autowired
  private ITimeSlotService timeSlotService;

  @Autowired
  private IUserService userService;

  @GetMapping(value = "/guest/time_slots/examination")
  public List<TimeSlotResponse> getAllTimeSlotByExaminationId(
      @RequestParam("examinationId") Long medicalExaminationId) {

    return timeSlotService.getAllTimeSlotByExaminationId(medicalExaminationId);
  }

  @GetMapping(value = "/guest/time_slots/{id}")
  public TimeSlotResponse getDetailTimeSlot(
      @PathVariable("id") Long timeSlotId) {
    return timeSlotService.getDetailTimeSlot(timeSlotId);
  }

  @GetMapping(value = "/time_slots/doctor")
  @PreAuthorize("hasAnyAuthority('ROLE_ADMIN', 'ROLE_DOCTOR')")
  public List<TimeSlotResponse> getAllTimeSlotOfDoctor() {
    UserResponse userResponse = userService.getCurrentUser();
    UserDTO doctor = userResponse.getUser();

    return timeSlotService.getAllTimeSlotsOfCurrentWeek(doctor.getId());
  }

  @PostMapping(value = "/time_slots")
  @PreAuthorize("hasAnyAuthority('ROLE_ADMIN', 'ROLE_DOCTOR')")
  public List<TimeSlotResponse> addAllTimeSlotOfDoctor(
      @RequestBody List<TimeSlotRequest> listTimeSlotRequest) {
    UserResponse userResponse = userService.getCurrentUser();
    UserDTO doctor = userResponse.getUser();

    return timeSlotService.addAllTimeSlotsOfDoctor(listTimeSlotRequest, doctor.getId());
  }

  @PatchMapping(path = "/time_slots/{id}")
  @PreAuthorize("hasAnyAuthority('ROLE_ADMIN', 'ROLE_DOCTOR')")
  public TimeSlotDTO updateTimeSlotByDoctor(
      @PathVariable(value = "id") Long timeSlotId,
      @RequestBody TimeSlotDTO timeSlotDTO) {
    UserResponse userResponse = userService.getCurrentUser();
    UserDTO doctor = userResponse.getUser();
    timeSlotDTO.setId(timeSlotId);

    return timeSlotService.updateTimeSlotByDoctor(timeSlotDTO, doctor.getId());
  }

  @DeleteMapping(path = "/time_slots/{id}")
  @PreAuthorize("hasAnyAuthority('ROLE_ADMIN', 'ROLE_DOCTOR')")
  public String deleteTimeSlotByDoctor(
      @PathVariable(value = "id") Long timeSlotId
  ) {
    UserResponse userResponse = userService.getCurrentUser();
    UserDTO doctor = userResponse.getUser();

    return timeSlotService.deleteTimeSlotByDoctor(timeSlotId);
  }
}
