import { Box } from '@mui/material'
import { useNavigate } from 'react-router-dom'

import '../../../assets/css/pages/guestPage/homePage/home_page.css'
import ButtonCustomize from '../../../components/ButtonCustomize'

const nurseImg = require('../../../assets/img/nurse-intro.png')

interface CardItemInterface {
  title: string
  content: string
  contentDetail: string
}

const cardItem: CardItemInterface[] = [
  {
    title: '15+',
    content: 'Years of ',
    contentDetail: 'Experience'
  },
  {
    title: '200',
    content: 'Doctors ',
    contentDetail: 'Speacialist'
  },
  {
    title: '100%',
    content: 'Patient ',
    contentDetail: 'Satisfaction'
  }
]

export default function HomePage() {
  const navigate = useNavigate()
  const handleClickBtn = () => {
    navigate('/search-doctor')
  }

  return (
    <Box className='home__page--container'>
      <Box className='home__banner--container'>
        <Box className='banner__intro--group'>
          <Box className='banner__intro--text'>
            <p className='banner__intro--title'>
              Find a doctor
              <span>for you now !</span>
            </p>
            <p className='banner__intor--des'>
              Discover the right doctor for you and your health needs - Find a Doctor now!
            </p>
          </Box>
          <Box className='banner__intro--btn'>
            <ButtonCustomize
              text='Get Appointment'
              className='btn__appointment btn__radius'
              onClickBtn={handleClickBtn}
            />
          </Box>
        </Box>
        <Box className='banner__intro--img'>
          <img src={nurseImg} alt='#' />
        </Box>
        <Box className='banner__intro--experience'>
          {cardItem.map((card) => {
            return (
              <Box className='box__intro--experience' key={card.title}>
                <p className='title--experience'>{card.title}</p>
                <p className='content--experience'>
                  {card.content}
                  <span>{card.contentDetail}</span>
                </p>
              </Box>
            )
          })}
        </Box>
      </Box>
    </Box>
  )
}
