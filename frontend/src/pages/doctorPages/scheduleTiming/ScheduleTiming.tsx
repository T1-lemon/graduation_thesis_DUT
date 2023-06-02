import TabContext from '@mui/lab/TabContext'
import TabList from '@mui/lab/TabList'
import Box from '@mui/material/Box'
import Tab from '@mui/material/Tab'
import React, { useEffect, useState } from 'react'
import { useDispatch, useSelector } from 'react-redux'

import '../../../assets/css/pages/doctorPage/scheduleTiming/schedule_timing.css'
import { ITimeSlotResponse } from '../../../interface/TimeSlotInterfaces'
import { DispatchType, RootState } from '../../../redux/configStore'
import { useAppSelector } from '../../../redux/hooks'
import { getAllTimeSlotOfDoctorThunk } from '../../../redux/slices/timeSlotSlice'
import {
  compareDate,
  getAllDateOfCurrentWeek,
  getDateOfWeek
} from '../../../utils/date'
import ModalSchedule from './ModalSchedule'
import TimeSlotItem from './TimeSlotItem'

type Props = {}

const arrDayOfWeek: string[] = getAllDateOfCurrentWeek()

export default function ScheduleTiming({}: Props) {
  const dispatch: DispatchType = useDispatch()
  const [value, setValue] = useState(arrDayOfWeek[0])

  const { arrTimeSlotOfDoctorInCurrentWeek } = useAppSelector(
    (state) => state.timeSlotsReducer
  )
  const { isCheckInitialStatus } = useAppSelector((state) => state.auths)
  const [timeSlotsOfDay, setTimeSlotsOfDay] = useState<ITimeSlotResponse[]>([])

  useEffect(() => {
    if (!isCheckInitialStatus) {
      const getAllTimeSlotOfDoctorInCurrentWeek = async () => {
        await dispatch(getAllTimeSlotOfDoctorThunk())
      }

      getAllTimeSlotOfDoctorInCurrentWeek()
    }
  }, [])

  const getTimeSlotItemOfCurrentDayOfWeek = () => {
    const arrTimeSlotsMonday = arrTimeSlotOfDoctorInCurrentWeek.filter(
      (timeSlot, index) => {
        return compareDate(value, timeSlot.timeSlotDTO.startTime)
      }
    )
    setTimeSlotsOfDay(arrTimeSlotsMonday)
  }

  useEffect(() => {
    getTimeSlotItemOfCurrentDayOfWeek()
  }, [
    arrTimeSlotOfDoctorInCurrentWeek.length,
    arrTimeSlotOfDoctorInCurrentWeek
  ])

  const handleChangeTab = (event: React.SyntheticEvent, newValue: string) => {
    const arrTimeSlots = arrTimeSlotOfDoctorInCurrentWeek.filter(
      (timeSlot, index) => {
        return compareDate(newValue, timeSlot.timeSlotDTO.startTime)
      }
    )
    setTimeSlotsOfDay(arrTimeSlots)
    setValue(newValue)
  }

  const renderTimeSlotItem = () => {
    if (arrTimeSlotOfDoctorInCurrentWeek.length === 0) return ''
    if (timeSlotsOfDay.length === 0) return <span>Not Available</span>
    timeSlotsOfDay.sort(function (a: ITimeSlotResponse, b: ITimeSlotResponse) {
      return (
        new Date(a.timeSlotDTO.startTime).getTime() -
        new Date(b.timeSlotDTO.startTime).getTime()
      )
    })
    return timeSlotsOfDay.map((item, index) => {
      return (
        <TimeSlotItem
          key={index}
          timeSlotResponse={item}
          timeSlotsOfDay={timeSlotsOfDay}
          dayOfWeek={new Date(value)}
        />
      )
    })
  }
  return (
    <div>
      <div className='container__schedule'>
        <div className='container__title'>
          <h3>Schedule Timing</h3>
        </div>

        <div className='schedule__widget'>
          <Box sx={{ width: '100%', typography: 'body1' }}>
            <TabContext value={value}>
              <Box className='schedule__header'>
                <TabList
                  onChange={handleChangeTab}
                  aria-label='lab API tabs example'
                  TabIndicatorProps={{
                    sx: {
                      backgroundColor: 'transparent'
                    }
                  }}
                  className='tab__list'
                >
                  {arrDayOfWeek.map((item, index) => {
                    return (
                      <Tab
                        key={index}
                        className='tab__header-item'
                        label={getDateOfWeek(item)}
                        value={item}
                      />
                    )
                  })}
                </TabList>
              </Box>

              <div className='tab__content-title'>
                <h3>Time Slots</h3>
                <ModalSchedule
                  dayOfWeek={value}
                  timeSlotsOfDay={timeSlotsOfDay}
                />
              </div>

              <div className='tab__content-times'>{renderTimeSlotItem()}</div>
            </TabContext>
          </Box>
        </div>
      </div>
    </div>
  )
}
