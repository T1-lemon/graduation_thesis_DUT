import CalendarMonthIcon from '@mui/icons-material/CalendarMonth'
import React, { useEffect, useState } from 'react'
import { NavLink } from 'react-router-dom'

import {
  ITimeSlot,
  ITimeSlotResponse
} from '../../../interface/TimeSlotInterfaces'
import { convertVND } from '../../../utils/convertMoney'
import {
  addHoursToDate,
  formatDate,
  getTimeSlotsAfterCurrentDay,
  getTimeSlotsDateFirst,
  getTimeZone,
  removeDuplicatesDate,
  sortDate,
  sortStartTime
} from '../../../utils/date'

type Props = {
  timeSlotsResponse?: ITimeSlotResponse[]
  examinationPrice?: number
}

export default function ScheduleExamination(props: Props) {
  const { timeSlotsResponse, examinationPrice } = props

  const timeSlotsNoAppointment = timeSlotsResponse
    ? timeSlotsResponse.filter((item) => !item.appointmentId)
    : []

  const timeSlots: ITimeSlot[] =
    timeSlotsNoAppointment.length === 0
      ? []
      : timeSlotsNoAppointment.map((item) => item.timeSlotDTO)

  const listTimeSlotsAfterCurrentDay: ITimeSlot[] = timeSlots
    ? getTimeSlotsAfterCurrentDay(timeSlots)
    : []

  const listDate: Date[] = listTimeSlotsAfterCurrentDay.map((item) => {
    let date = new Date(item.startTime)
    date.setHours(0, 0, 0, 0)
    return date
  })

  const listDateSort: Date[] = sortDate(removeDuplicatesDate(listDate))

  const listTimeSlotsCurrentDay: ITimeSlot[] = getTimeSlotsDateFirst(
    listTimeSlotsAfterCurrentDay,
    listDateSort[0]
  )

  const [listTimeSlot, setListTimeSlot] = useState(listTimeSlotsCurrentDay)

  useEffect(() => {
    setListTimeSlot(sortStartTime(listTimeSlotsCurrentDay))
  }, [timeSlotsResponse])

  const handleDateSelect = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const dateSelect: string = event.target.value

    const listTimeOfDateSelect = listTimeSlotsAfterCurrentDay.filter((item) => {
      const dateItem = new Date(item.startTime)
      dateItem.setHours(0, 0, 0, 0)
      return dateItem.getTime() === new Date(dateSelect).getTime()
    })

    const listTimeSlotSort = sortStartTime(listTimeOfDateSelect)

    setListTimeSlot(listTimeSlotSort)
  }

  return (
    <div className='container__schedule'>
      <div className='title__box'>
        <h3>Book Appointment</h3>
      </div>
      <div className='box__schedule'>
        <select
          name=''
          id=''
          className='schedule__select'
          onChange={handleDateSelect}
        >
          {listDateSort.map((date, index) => {
            return (
              <option key={index} value={date.toString()}>
                {formatDate(date)}
              </option>
            )
          })}
        </select>

        <div className='container__schedule--package'>
          <div className='container__schedule--icon'>
            <CalendarMonthIcon />
            <b>SCHEDULE</b>
          </div>
          <div className='container__schedule__time'>
            {listTimeSlot.map((item, index) => {
              return (
                <NavLink key={index} to={`/book-appointment/${item.id}`}>
                  <span className='schedule__time--item'>
                    {`${getTimeZone(item.startTime)} - ${getTimeZone(
                      addHoursToDate(new Date(item.startTime), item.duration)
                    )}`}
                  </span>
                </NavLink>
              )
            })}
          </div>
        </div>
      </div>
      <div className='examination__price'>
        <div>
          <b>Examination Price: </b>
          <span className='value__price'>
            {examinationPrice ? convertVND.format(examinationPrice) : ''}
          </span>
        </div>

        <div>
          <b>Booking fee: </b>
          <span className='value__price'>Free</span>
        </div>
      </div>
    </div>
  )
}
