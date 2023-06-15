package com.ces.hospitalcare.service;
import com.ces.hospitalcare.dto.AppointmentDTO;
import com.ces.hospitalcare.http.request.AppointmentRequest;
import com.ces.hospitalcare.http.response.AppointmentResponse;
import com.ces.hospitalcare.http.response.CheckResponse;
import java.util.List;
import org.springframework.data.domain.Pageable;

public interface IAppointmentService {
  int countAppointment();

  String cancelAppointmentMedicalArchive(Long doctorId);

  List<AppointmentResponse> getListAppointmentOfPatient();

  int countByStatusAndDoctorId(int status, Long doctorId);

  List<AppointmentDTO> getAllAppointmentOfDoctorPageableByStatusAndDoctorId(int appointmentStatus,
      Long doctorId, Pageable pageable);

  List<AppointmentDTO> getAllAppointmentPageable(Pageable pageable);

  AppointmentDTO getDetailAppointment(Long appointmentId);

  List<AppointmentDTO> getAllByDoctorIdAndPatientId(Long doctorId, Long patientId);

  AppointmentDTO changeStatusByDoctor(AppointmentDTO appointmentChangeStatusDTO, Long doctorId);

  AppointmentDTO bookAppointmentByPatient(AppointmentRequest appointmentRequest);

  CheckResponse checkAppointmentByPatientAndExamination(Long patientId, Long examinationId);
}
