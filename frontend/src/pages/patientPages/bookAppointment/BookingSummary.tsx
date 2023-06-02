import React from 'react'

import { ITimeSlot } from '../../../interface/TimeSlotInterfaces'
import { convertVND } from '../../../utils/convertMoney'
import { addHoursToDate, formatDate, getTimeZone } from '../../../utils/date'

type Props = {
  timeSlot?: ITimeSlot
}

export default function BookingSummary(props: Props) {
  const { timeSlot } = props

  const arrInforSummary = [
    {
      title: 'Doctor',
      value: timeSlot?.medicalExamination.title
    },
    {
      title: 'Date',
      value: timeSlot ? formatDate(new Date(timeSlot.startTime)) : ''
    },
    {
      title: 'Time',
      value: timeSlot
        ? `${getTimeZone(timeSlot.startTime)} - ${getTimeZone(
            addHoursToDate(new Date(timeSlot.startTime), timeSlot.duration)
          )}`
        : ''
    }
  ]

  return (
    <div className='booking__summary container__form__book'>
      <div className='title__box'>
        <h3 className='form__title'>Booking Summary</h3>
      </div>

      <div className='inner__box'>
        <div className='information__summary'>
          <ul>
            {arrInforSummary.map((item, index) => {
              return (
                <li key={index}>
                  <p className='title__information'>{item.title}</p>
                  <span className='value'>{item.value}</span>
                </li>
              )
            })}
          </ul>
        </div>

        <div className='information__price'>
          <div>
            Examination Price
            <span className='value'>
              {' '}
              {timeSlot
                ? convertVND.format(
                    timeSlot.medicalExamination.examinationPrice
                  )
                : ''}
            </span>
          </div>

          <div>
            <b>Booking fee </b>
            <span className='value'>Free</span>
          </div>
        </div>

        <div className='container__total__price'>
          <h3>Total</h3>
          <span className='value total__price'>
            {timeSlot
              ? convertVND.format(timeSlot.medicalExamination.examinationPrice)
              : ''}
          </span>
        </div>

        <div className='payment'>
          <h3>Payment</h3>
          <span className='value'>Pay later at a medical facility</span>
        </div>
      </div>
    </div>
  )
}
