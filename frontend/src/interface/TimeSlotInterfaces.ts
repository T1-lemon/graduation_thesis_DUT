import { IMedicalExamination } from './MedicalExaminationInterfaces'

export interface ITimeSlot {
  id: number
  createdBy: null
  modifiedBy: null
  createdDate: null
  modidiedDate: null
  startTime: string
  duration: number
  medicalExamination: IMedicalExamination
}

export interface ITimeSlotResponse {
  timeSlotDTO: ITimeSlot
  doctorId: number
  appointmentId: number
}

export interface ITimeSlotRequest {
  startTime: string
  duration: number
}
