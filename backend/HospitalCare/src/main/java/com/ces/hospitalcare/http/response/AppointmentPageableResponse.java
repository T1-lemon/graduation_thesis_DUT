package com.ces.hospitalcare.http.response;
import com.ces.hospitalcare.dto.AppointmentDTO;
import java.util.ArrayList;
import java.util.List;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class AppointmentPageableResponse {
  private int pageIndex;

  private int totalPage;

  private List<AppointmentDTO> listAppointmentResult = new ArrayList<>();
}
