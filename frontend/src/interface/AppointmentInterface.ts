import { ITimeSlot } from './TimeSlotInterfaces'
import { UserInformation } from './UsersInterface'

export interface AppointmentInforInterface {
  duration: number
  startTime: string
  examinationPrice: number
  doctorId: number
  lastNameDoctor: string
  genderDoctor: boolean
  profilePictureDoctor: string
  nameDepartment: string
  firstNameDoctor: string
  appointmentId: number
  patientId: number
  status: number
  modifiedDate: Date
}

export interface IAppointment {
  id: number
  createdBy: null
  modifiedBy: null
  createdDate: Date
  modifiedDate: Date
  status: number
  patient: UserInformation
  timeSlot: ITimeSlot
}

export interface IAppointmentPageable {
  pageIndex: number
  totalPage: number
  listAppointmentResult: IAppointment[]
}

export interface ICheckAppointment {
  statusCode: number
  message: string
}
