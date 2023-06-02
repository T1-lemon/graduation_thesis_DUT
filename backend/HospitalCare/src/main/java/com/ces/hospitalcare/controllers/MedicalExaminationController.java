package com.ces.hospitalcare.controllers;
import com.ces.hospitalcare.dto.MedicalExaminationDTO;
import com.ces.hospitalcare.http.request.MedicalExaminationRequest;
import com.ces.hospitalcare.http.response.MedicalExaminationResponse;
import com.ces.hospitalcare.service.IMedicalExaminationService;
import java.util.List;
import org.springframework.beans.factory.annotation.Autowired;
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
@RequestMapping(path = "/api/v1")
public class MedicalExaminationController {
  @Autowired
  private IMedicalExaminationService medicalExaminationService;

  @GetMapping(path = "/guest/medical_examinations")
  public List<MedicalExaminationResponse> getAllMedicalExamination() {

    return medicalExaminationService.getAllMedicalExamination();
  }

  @GetMapping(value = "/guest/medical_examinations/{id}")
  public MedicalExaminationResponse getDetailMedicalExamination(
      @PathVariable("id") Long medicalExaminationId) {
    return medicalExaminationService.getDetailMedicalExamination(medicalExaminationId);
  }

  @GetMapping(value = "/guest/medical_examinations/filter")
  public List<MedicalExaminationResponse> filterMedicalExaminationByCategoryAndPrice(
      @RequestParam(value = "category[]") String[] categories,
      @RequestParam(value = "minPrice", required = false) Long minPrice,
      @RequestParam(value = "maxPrice", required = false) Long maxPrice) {

    return medicalExaminationService.filterMedicalExaminationByCategoryAndPrice(
        minPrice, maxPrice, categories);
  }

  @PatchMapping(path = "/medical_examinations/archive/{id}")
  @PreAuthorize("hasAnyAuthority('ROLE_ADMIN')")
  public MedicalExaminationDTO archiveMedicalExamination(
      @PathVariable(value = "id") Long medicalExaminationId,
      @RequestBody MedicalExaminationDTO medicalExaminationDTO) {

    medicalExaminationDTO.setId(medicalExaminationId);

    return medicalExaminationService.archiveMedicalExamination(medicalExaminationDTO);
  }

  @PatchMapping(path = "/medical_examinations/{id}")
  @PreAuthorize("hasAnyAuthority('ROLE_ADMIN')")
  public MedicalExaminationDTO updateMedicalExamination(
      @PathVariable(value = "id") Long medicalExaminationId,
      @RequestBody MedicalExaminationRequest medicalExaminationRequest) {

    return medicalExaminationService.updateMedicalExamination(medicalExaminationRequest,
        medicalExaminationId);
  }

  @PostMapping(path = "/medical_examinations")
  @PreAuthorize("hasAnyAuthority('ROLE_ADMIN')")
  public MedicalExaminationDTO addMedicalExamination(

      @RequestBody MedicalExaminationRequest medicalExaminationRequest) {

    return medicalExaminationService.addMedicalExamination(medicalExaminationRequest);
  }
}
