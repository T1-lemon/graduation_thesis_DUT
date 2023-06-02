package com.ces.hospitalcare.dto;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class DepartmentDTO extends BaseDTO {
  private String name;

  private String backgroundImage;
}
