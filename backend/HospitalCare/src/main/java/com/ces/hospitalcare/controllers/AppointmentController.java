package com.ces.hospitalcare.controllers;
import com.ces.hospitalcare.dto.AppointmentDTO;
import com.ces.hospitalcare.dto.UserDTO;
import com.ces.hospitalcare.http.request.AppointmentRequest;
import com.ces.hospitalcare.http.response.AppointmentPageableResponse;
import com.ces.hospitalcare.http.response.AppointmentResponse;
import com.ces.hospitalcare.http.response.UserResponse;
import com.ces.hospitalcare.service.IAppointmentService;
import com.ces.hospitalcare.service.IUserService;
import java.util.List;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/appointments")
public class AppointmentController {
  @Autowired
  private IAppointmentService appointmentService;

  @Autowired
  private IUserService userService;

  @GetMapping("/patient")
  @PreAuthorize("hasAuthority('ROLE_PATIENT')")
  public ResponseEntity<List<AppointmentResponse>> getListAppointmentOfPatient() {
    List<AppointmentResponse> listAppointmentOfPatient = appointmentService.getListAppointmentOfPatient();
    return ResponseEntity.status(HttpStatus.OK).body(listAppointmentOfPatient);
  }

  @GetMapping(path = "/doctor/pageable")
  @PreAuthorize("hasAnyAuthority('ROLE_ADMIN', 'ROLE_DOCTOR')")
  public AppointmentPageableResponse getAllAppointmentOfDoctorPageableByStatusAndDoctorId(
      @RequestParam("pageIndex") int pageIndex, @RequestParam("limit") int limit,
      @RequestParam("appointmentStatus") int appointmentStatus) {

    UserResponse userResponse = userService.getCurrentUser();
    UserDTO userDTO = userResponse.getUser();

    Pageable pageable = PageRequest.of(pageIndex - 1, limit);
    int totalPage = (int) Math.ceil(
        (double) (appointmentService.countByStatusAndDoctorId(appointmentStatus, userDTO.getId()))
            / limit);
    List<AppointmentDTO> listAppointmentDTO = appointmentService.getAllAppointmentOfDoctorPageableByStatusAndDoctorId(
        appointmentStatus, userDTO.getId(), pageable);
    return AppointmentPageableResponse.builder().pageIndex(pageIndex).totalPage(totalPage)
        .listAppointmentResult(listAppointmentDTO).build();
  }

  @GetMapping(path = "/doctor/patient")
  @PreAuthorize("hasAnyAuthority('ROLE_ADMIN', 'ROLE_DOCTOR')")
  public List<AppointmentDTO> getAllByDoctorIdAndPatientId(
      @RequestParam(value = "patientId") Long patientId) {
    UserResponse userResponse = userService.getCurrentUser();
    UserDTO userDTO = userResponse.getUser();
    return appointmentService.getAllByDoctorIdAndPatientId(userDTO.getId(), patientId);
  }

  @GetMapping(path = "/pageable")
  @PreAuthorize("hasAnyAuthority('ROLE_ADMIN')")
  public AppointmentPageableResponse getAllAppointmentPageable(
      @RequestParam("pageIndex") int pageIndex, @RequestParam("limit") int limit) {

    Pageable pageable = PageRequest.of(pageIndex - 1, limit);
    int totalPage = (int) Math.ceil((double) (appointmentService.countAppointment()) / limit);
    List<AppointmentDTO> listAppointmentDTO = appointmentService.getAllAppointmentPageable(
        pageable);
    return AppointmentPageableResponse.builder().pageIndex(pageIndex).totalPage(totalPage)
        .listAppointmentResult(listAppointmentDTO).build();
  }

  @GetMapping(path = "/{id}")
  @PreAuthorize("hasAnyAuthority('ROLE_ADMIN')")
  public AppointmentDTO getDetailAppointment(
      @PathVariable(value = "id") Long appointmentId) {

    return appointmentService.getDetailAppointment(appointmentId);
  }

  @PatchMapping(path = "/change_status_appointment/{id}")
  @PreAuthorize("hasAnyAuthority('ROLE_ADMIN', 'ROLE_DOCTOR')")
  public AppointmentDTO changeStatusAppointmentByDoctor(
      @PathVariable(value = "id") Long appointmentId, @RequestBody AppointmentDTO appointmentDTO) {
    UserResponse userResponse = userService.getCurrentUser();
    UserDTO userDTO = userResponse.getUser();
    appointmentDTO.setId(appointmentId);

    return appointmentService.changeStatusByDoctor(appointmentDTO, userDTO.getId());
  }

  @PostMapping(path = "/book_appointment")
  @PreAuthorize("hasAnyAuthority('ROLE_ADMIN', 'ROLE_PATIENT')")
  public AppointmentDTO bookAppointmentByPatient(
      @RequestBody AppointmentRequest appointmentRequest) {
    UserResponse userResponse = userService.getCurrentUser();
    UserDTO patient = userResponse.getUser();
    appointmentRequest.setPatientId(patient.getId());
    return appointmentService.bookAppointmentByPatient(appointmentRequest);
  }
}
