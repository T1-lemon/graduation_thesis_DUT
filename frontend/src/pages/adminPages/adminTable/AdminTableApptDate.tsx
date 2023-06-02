import React from 'react'

type Props = {
  apptDate: string
  time: string
}

const AdminTableApptDate = (props: Props) => {
  const { apptDate, time } = props
  return (
    <div className='admin__table--apptDate'>
      <p className='table__p--apptDate'>{apptDate}</p>
      <p className='table__p--time'>{time}</p>
    </div>
  )
}

export default AdminTableApptDate
