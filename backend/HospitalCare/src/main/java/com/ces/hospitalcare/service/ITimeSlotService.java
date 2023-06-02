package com.ces.hospitalcare.service;
import com.ces.hospitalcare.dto.TimeSlotDTO;
import com.ces.hospitalcare.http.request.TimeSlotRequest;
import com.ces.hospitalcare.http.response.TimeSlotResponse;
import java.util.List;

public interface ITimeSlotService {
  List<TimeSlotResponse> getAllTimeSlotByExaminationId(Long examinationId);

  TimeSlotResponse getDetailTimeSlot(Long id);

  List<TimeSlotResponse> getAllTimeSlotsOfCurrentWeek(Long doctorId);

  List<TimeSlotResponse> addAllTimeSlotsOfDoctor(List<TimeSlotRequest> listTimeSlotRequest,
      Long doctorId);

  TimeSlotDTO updateTimeSlotByDoctor(TimeSlotDTO timeSlotDTO, Long doctorId);

  String deleteTimeSlotByDoctor(Long timeSlotID);
}
