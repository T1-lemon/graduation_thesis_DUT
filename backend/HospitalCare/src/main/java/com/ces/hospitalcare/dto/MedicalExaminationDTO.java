package com.ces.hospitalcare.dto;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class MedicalExaminationDTO extends BaseDTO {
  private Long examinationPrice;

  private String title;

  private String shortDescription;

  private String description;

  private String image;

  private Integer statusArchive;

  private DepartmentDTO department;
}
