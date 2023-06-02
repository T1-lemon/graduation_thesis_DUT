import TableBookedAppointment from './TableBookedAppointment'

const PatientBookedAppoitment = () => {
  return (
    <div className='appointment__container'>
      <div className='appointment__div--header'>
        <p className='appointment__p--title'>Appointments</p>
        <p className='appointment__p--des'>
          You can check your appointments here
        </p>
      </div>
      <div className='appointment__div--body'>
        <TableBookedAppointment />
      </div>
    </div>
  )
}

export default PatientBookedAppoitment
